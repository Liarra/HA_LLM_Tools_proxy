"""Minimal FastAPI application that mimics the OpenAI Chat Completions endpoint.
It logs every incoming request body to a uniquely‑named text file
   (<model>_<timestamp>.txt) and replies with a fixed assistant message
   saying "noted", formatted according to the OpenAI protocol.

Run with:
    uvicorn fastapi_openai_proxy:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import datetime as _dt
import json
import json as _json
import os as _os
import uuid as _uuid
from pathlib import Path as _Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from count_tokens.count import count_tokens_in_string

from tools_storage import get_3_random_tools, dump_tools_into_storage

# Directory in which to store request logs (created automatically)
_LOG_DIR = _Path("logs_dest")
_LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Minimal OpenAI‑like API")


def _current_timestamp() -> str:
    """Return an ISO‑like timestamp safe for filenames (UTC)."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    """Count the number of tokens in a text string."""
    return count_tokens_in_string(text)

def get_list_of_tools(request_body: dict) -> list:
    """Extract the list of tools from the request body."""
    tools = request_body.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("Tools should be a list.")
    return tools

def get_user_message(request_body: dict) -> str:
    """Extract the user message from the request body."""
    messages = request_body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Messages should be a list.")

    last_message=None
    for message in messages:
        if message.get("role") == "user":
            last_message=message

    if last_message is not None:
        return last_message["content"]

    raise ValueError("No user message found in the request body.")

def get_system_message(request_body: dict) -> str:
    """Extract the system message from the request body."""
    messages = request_body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Messages should be a list.")

    for message in messages:
        if message.get("role") == "system":
            return message.get("content", "")

    return ""  # Return empty string if no system message is found

def strip_tools_from_request(request_body: dict) -> dict:
    """Remove the 'tools' key from the request body."""
    if "tools" in request_body:
        del request_body["tools"]
        request_body["tools"]=[]
    return request_body

def add_tools_to_request(request_body: dict, tools: list) -> dict:
    """Add a list of tools to the request body."""
    if not isinstance(tools, list):
        raise ValueError("Tools should be a list.")
    request_body["tools"] = tools
    return request_body

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle OpenAI‑style chat.completions requests."""
    body = await request.json()
    model: str = body.get("model", "unknown-model")

    # ---- Print User Message ----
    user_message = get_user_message(body)
    print("-"*10)
    print(f"User message: {user_message}")
    print("-"*10)

    # ---- Print messages stats ----
    system_message = get_system_message(body)
    tools_dict = get_list_of_tools(body)
    tools_str = json.dumps(tools_dict)

    print("Tokens in System message:", count_tokens(system_message))
    print("Tokens in User message:", count_tokens(user_message))
    print("Tokens in Tools:", count_tokens(tools_str))
    print("-"*10)

    # ---- Logging ----
    filename = f"{model}_{_current_timestamp()}.txt"
    log_path = _LOG_DIR / filename
    with log_path.open("w", encoding="utf-8") as fp:
        _json.dump(body, fp, ensure_ascii=False, indent=2)

    # ---- Stub completion ----
    resp = {
        "id": f"chatcmpl-{_uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(_dt.datetime.utcnow().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sure, sure. I wrote all of that down!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "total_tokens": 1,
        },
    }

    return JSONResponse(content=resp)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("dummy:app", host="0.0.0.0", port=8001, reload=True)
