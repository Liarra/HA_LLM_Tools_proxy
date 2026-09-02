"""OpenAI-compatible reverse proxy with semantic tool filtering."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import dotenv
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from embedding import SemanticEmbedder
from tools_storage import Selection, ToolSelector, tool_name

dotenv.load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _names_from_env(name: str, default: Sequence[str]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return tuple(default)
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    upstream_base_url: str = "https://api.openai.com/v1"
    upstream_api_key: str = ""
    max_tools: int = 3
    min_tool_similarity: float = 0.75
    whitelisted_tools: tuple[str, ...] = ("GetLiveContext",)
    blacklisted_tools: tuple[str, ...] = (
        "HassHumidifierMode",
        "HassHumidifierSetPoint",
    )
    embedding_model: str = "intfloat/e5-small-v2"
    embedding_cache: Path = Path("data/tool_embeddings.sqlite3")

    @classmethod
    def from_env(cls) -> AppConfig:
        """Create validated configuration from the process environment."""
        return cls(
            upstream_base_url=os.getenv(
                "OPENAI_API_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            upstream_api_key=os.getenv("OPENAI_API_KEY", ""),
            max_tools=int(os.getenv("TOOLS_TO_KEEP", "3")),
            min_tool_similarity=float(os.getenv("MIN_TOOL_SIMILARITY", "0.75")),
            whitelisted_tools=_names_from_env("WHITELISTED_TOOLS", ("GetLiveContext",)),
            blacklisted_tools=_names_from_env(
                "BLACKLISTED_TOOLS",
                ("HassHumidifierMode", "HassHumidifierSetPoint"),
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "intfloat/e5-small-v2"),
            embedding_cache=Path(
                os.getenv("EMBEDDING_CACHE", "data/tool_embeddings.sqlite3")
            ),
        )


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


def selection_query(body: dict) -> str:
    """Use the latest user turn, with assistant context for very short replies."""
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return ""

    last_user = ""
    last_assistant = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = _content_text(message.get("content"))
        if message.get("role") == "user":
            last_user = content
        elif message.get("role") == "assistant":
            last_assistant = content

    if len(last_user.split()) < 4 and last_assistant:
        return f"{last_assistant} ... {last_user}"
    return last_user


def _required_tool_names(body: dict) -> tuple[str, ...]:
    choice = body.get("tool_choice")
    if not isinstance(choice, dict):
        return ()
    function = choice.get("function", {})
    name = function.get("name") if isinstance(function, dict) else None
    return (name,) if isinstance(name, str) and name else ()


def _request_headers(request: Request, api_key: str) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS | {"host", "content-length"}
    }
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _HOP_BY_HOP_HEADERS | {"content-length"}
    }


async def _close_after_stream(response: httpx.Response):
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()


def create_app(
    *,
    config: AppConfig | None = None,
    selector: ToolSelector | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy application; dependency arguments make it testable."""
    config = config or AppConfig.from_env()
    selector = selector or ToolSelector(
        max_tools=config.max_tools,
        min_similarity=config.min_tool_similarity,
        whitelisted_names=config.whitelisted_tools,
        blacklisted_names=config.blacklisted_tools,
        embedder=SemanticEmbedder(config.embedding_model),
        cache_path=config.embedding_cache,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        timeout = httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
        application.state.upstream = httpx.AsyncClient(
            timeout=timeout, transport=transport
        )
        yield
        await application.state.upstream.aclose()

    application = FastAPI(title="HA LLM Tools Proxy", lifespan=lifespan)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.api_route(
        "/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
    )
    async def proxy(path: str, request: Request):
        raw_body = await request.body()
        stream = False

        if path == "chat/completions" and request.method == "POST":
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                return JSONResponse({"detail": "Request body must be valid JSON"}, 400)

            stream = body.get("stream") is True
            tools = body.get("tools")
            if isinstance(tools, list) and tools:
                try:
                    selected: Selection = await asyncio.to_thread(
                        selector.select,
                        tools,
                        selection_query(body),
                        required_names=_required_tool_names(body),
                        require_at_least_one=body.get("tool_choice") == "required",
                    )
                    body["tools"] = selected.tools
                    if not selected.tools:
                        body.pop("tools")
                        if body.get("tool_choice") == "auto":
                            body["tool_choice"] = "none"
                    logger.info(
                        "Selected %d/%d tools (max=%d, threshold=%.3f): %s",
                        len(selected.tools),
                        len(tools),
                        config.max_tools,
                        config.min_tool_similarity,
                        ", ".join(tool_name(tool) for tool in selected.tools) or "none",
                    )
                    logger.debug("Semantic scores: %s", selected.scores)
                except Exception:
                    logger.exception(
                        "Tool selection failed; forwarding all request tools"
                    )
            raw_body = json.dumps(body, ensure_ascii=False).encode("utf-8")

        query = request.url.query
        upstream_url = f"{config.upstream_base_url}/{path}"
        if query:
            upstream_url = f"{upstream_url}?{query}"
        upstream_request = request.app.state.upstream.build_request(
            request.method,
            upstream_url,
            headers=_request_headers(request, config.upstream_api_key),
            content=raw_body,
        )
        upstream_response = await request.app.state.upstream.send(
            upstream_request, stream=True
        )
        response_headers = _response_headers(upstream_response)

        if stream:
            return StreamingResponse(
                _close_after_stream(upstream_response),
                status_code=upstream_response.status_code,
                headers=response_headers,
                background=BackgroundTask(upstream_response.aclose),
            )

        try:
            content = await upstream_response.aread()
        finally:
            await upstream_response.aclose()
        return Response(
            content=content,
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    return application


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("front:app", host="0.0.0.0", port=8000)
