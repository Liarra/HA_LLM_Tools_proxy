"""Exercise the same LiteLLM Router streaming path as Custom Conversation."""

from __future__ import annotations

import asyncio
import json
import socket
from threading import Event, Thread

import httpx
import pytest
import uvicorn
from litellm import Router

from front import AppConfig, create_app
from tools_storage import Selection

pytestmark = pytest.mark.compatibility


class StubSelector:
    def select(self, tools, query, **kwargs) -> Selection:
        return Selection([tools[0]], {"GetLiveContext": 1.0})


class AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes, release: Event | None = None) -> None:
        self.chunks = chunks
        self.release = release

    async def __aiter__(self):
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if index == 0 and self.release is not None:
                released = await asyncio.to_thread(self.release.wait, 2)
                if not released:
                    raise TimeoutError("Proxy buffered the upstream SSE response")


def tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


async def _custom_conversation_request(base_url: str, release: Event):
    router = Router(
        model_list=[
            {
                "model_name": "qwen3",
                "litellm_params": {
                    "model": "openai/qwen3",
                    "api_base": f"{base_url}/v1",
                    "api_key": "home-assistant-test-key",
                },
            }
        ]
    )
    stream = await router.acompletion(
        model="qwen3",
        messages=[{"role": "user", "content": "What is the temperature?"}],
        tools=[tool("GetLiveContext"), tool("TurnOnLight")],
        stream=True,
        stream_options={"include_usage": True},
    )
    first_chunk = await stream.__anext__()
    release.set()
    chunks = [first_chunk, *[chunk async for chunk in stream]]
    return chunks


def test_custom_conversation_litellm_streaming_contract() -> None:
    seen: dict = {}
    release = Event()

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        first_event = (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen3","choices":[{"index":0,'
            '"delta":{"role":"assistant","content":"OK"},"finish_reason":null}]}\n\n'
        )
        remaining_events = (
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen3","choices":[{"index":0,"delta":{},'
            '"finish_reason":"stop"}],"usage":null}\n\n'
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk",'
            '"created":1,"model":"qwen3","choices":[],"usage":'
            '{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            stream=AsyncBytes(
                first_event.encode(), remaining_events.encode(), release=release
            ),
            headers={"content-type": "text/event-stream"},
        )

    app = create_app(
        config=AppConfig(upstream_base_url="http://upstream.test/v1"),
        selector=StubSelector(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(upstream),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        thread.join(0.05)

    try:
        chunks = asyncio.run(
            _custom_conversation_request(f"http://127.0.0.1:{port}", release)
        )
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert (
        "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        )
        == "OK"
    )
    assert any(
        (usage := getattr(chunk, "usage", None)) and usage.total_tokens == 11
        for chunk in chunks
    )
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert [item["function"]["name"] for item in seen["body"]["tools"]] == [
        "GetLiveContext"
    ]
