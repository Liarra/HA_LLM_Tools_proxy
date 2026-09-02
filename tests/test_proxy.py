from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from front import AppConfig, create_app
from tools_storage import Selection


def tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class StubSelector:
    def __init__(self, selected: list[dict]) -> None:
        self.selected = selected

    def select(self, tools, query, **kwargs) -> Selection:
        return Selection(self.selected, {"SelectedTool": 0.9})


class AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def config() -> AppConfig:
    return AppConfig(
        upstream_base_url="http://upstream.test/v1",
        upstream_api_key="upstream-secret",
        max_tools=3,
        min_tool_similarity=0.75,
    )


def test_empty_environment_list_disables_default_whitelist(monkeypatch) -> None:
    monkeypatch.setenv("WHITELISTED_TOOLS", "")

    assert AppConfig.from_env().whitelisted_tools == ()


def test_models_endpoint_is_transparent() -> None:
    seen: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json={"object": "list", "data": []})

    app = create_app(
        config=config(),
        selector=StubSelector([]),  # type: ignore[arg-type]
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.get("/v1/models?source=ha")

    assert response.json() == {"object": "list", "data": []}
    assert seen == {
        "url": "http://upstream.test/v1/models?source=ha",
        "authorization": "Bearer upstream-secret",
    }


def test_non_streaming_chat_is_filtered_and_status_is_preserved() -> None:
    selected = tool("SelectedTool")

    def upstream(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"] == [selected]
        return httpx.Response(429, json={"error": {"message": "busy"}})

    app = create_app(
        config=config(),
        selector=StubSelector([selected]),  # type: ignore[arg-type]
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "turn on the light"}],
                "tools": [selected, tool("OtherTool")],
            },
        )

    assert response.status_code == 429
    assert response.json() == {"error": {"message": "busy"}}


def test_zero_selected_tools_are_omitted_from_upstream_request() -> None:
    seen: dict = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    app = create_app(
        config=config(),
        selector=StubSelector([]),  # type: ignore[arg-type]
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello there"}],
                "tools": [tool("OtherTool")],
                "tool_choice": "auto",
            },
        )

    assert response.status_code == 200
    assert "tools" not in seen
    assert seen["tool_choice"] == "none"


def test_sse_stream_is_forwarded_from_correct_endpoint() -> None:
    seen: dict = {}
    event_stream = (
        b'data: {"id":"one","choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n'
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            stream=AsyncBytes(event_stream[:30], event_stream[30:]),
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(
        config=config(),
        selector=StubSelector([]),  # type: ignore[arg-type]
        transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == event_stream
    assert seen["url"] == "http://upstream.test/v1/chat/completions"
    assert seen["body"]["stream"] is True
