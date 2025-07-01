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
import os as _os
from pathlib import Path as _Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import dotenv
import httpx
import uvicorn

from count_tokens.count import count_tokens_in_string

from tools_storage import dump_tools_into_storage, get_n_most_relevant_tools, load_tools, save_tools

dotenv.load_dotenv()
_TOOLS_TO_KEEP=int(_os.getenv("TOOLS_TO_KEEP", "3"))

# Directory in which to store request logs (created automatically)
_LOG_DIR = _Path("logs")
_DATA_DIR = _Path("data")
_DATA_DIR.mkdir(exist_ok=True)
_LOG_DIR.mkdir(exist_ok=True)

_STORAGE_DB = _Path("data/tools_storage.sqlite3")

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

def get_last_assistant_message(request_body: dict) -> str:
    """Extract the last assistant message from the request body."""
    messages = request_body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("Messages should be a list.")

    last_message=None
    for message in messages:
        if message.get("role") == "assistant":
            last_message=message

    if last_message is not None:
        return last_message["content"]

    return "" # Return empty string if no assistant message is found

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

def print_stats(message_body:dict):
    """Print statistics about the request body."""
    print("-"*10)
    print("Tokens in System message:", count_tokens(get_system_message(message_body)))
    print("Tokens in User message:", count_tokens(get_user_message(message_body)))
    tools_dict = get_list_of_tools(message_body)
    tools_str = str(tools_dict)
    print("Tokens in Tools:", count_tokens(tools_str))
    print("-"*10)

def add_descriptions_if_missing(tools: list) -> list:
    """Add descriptions to tools if they are missing."""
    for tool in tools:
        if "function" in tool and "description" not in tool["function"]:
            tool["function"]["description"] = tool["function"].get("name", "No description available")
    return tools

def optimise_tools(message_body: dict) -> dict:
    """Optimise the tools in the request body."""
    tools_dict = get_list_of_tools(message_body)
    dump_tools_into_storage(tools_dict)
    new_body = strip_tools_from_request(message_body)

    user_request=get_user_message(new_body)

    words_user_request = len(user_request.split())
    if words_user_request < 4:
        # If the user request is too short, try to prepend the last assistant message for context
        last_assistant_message = get_last_assistant_message(new_body)
        if last_assistant_message is not None and len(last_assistant_message) > 0:
            print("User request is too short, prepending last assistant message for context.")
            user_request=last_assistant_message+" ... " +user_request

    new_tools = get_n_most_relevant_tools(user_request, n=_TOOLS_TO_KEEP)
    new_tools_openai_format=add_descriptions_if_missing(new_tools)

    print("Tokens in new Tools:", count_tokens(str(new_tools_openai_format)))

    new_body = add_tools_to_request(new_body, new_tools_openai_format)
    return new_body


def forward_request_to(request_body: dict, url: str, api_key: str, stream: bool = False):
    """Forward the request body to another URL and return the response."""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if stream:
        with httpx.stream("POST", url, json=request_body, headers=headers) as response:
            if response.status_code != 200:
                raise ValueError(f"Failed to forward request: {response.status_code} {response.text}")
            # Return the streaming response directly
            return response
    else:
        response = httpx.post(url, json=request_body, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Failed to forward request: {response.status_code} {response.text}")
        return response.json()


@app.on_event("startup")
async def _on_startup():
    if _STORAGE_DB.exists():
        print(f"Loading tools storage from {_STORAGE_DB}")
        load_tools(str(_STORAGE_DB))


@app.on_event("shutdown")
async def _on_shutdown():
    print(f"Saving tools storage to {_STORAGE_DB}")
    save_tools(str(_STORAGE_DB))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle OpenAI‑style chat.completions requests."""
    body = await request.json()
    model: str = body.get("model", "unknown-model")
    stream = body.get("stream", False)

    # ---- Print User Message ----
    user_message = get_user_message(body)
    print("-"*10)
    print(f"User message: {user_message}")
    print("-"*10)

    # ---- Print messages stats ----
    print_stats(body)

    # ---- Logging ----
    # filename = f"{model}_{_current_timestamp()}.txt"
    # log_path = _LOG_DIR / filename
    # with log_path.open("w", encoding="utf-8") as fp:
    #     _json.dump(body, fp, ensure_ascii=False, indent=2)

    # ---- Strip all tools and add 3 random tools ----
    new_body = optimise_tools(body)

    # filename = f"REWRITTEN_{model}_{_current_timestamp()}.txt"
    # log_path = _LOG_DIR / filename
    # with log_path.open("w", encoding="utf-8") as fp:
    #     _json.dump(new_body, fp, ensure_ascii=False, indent=2)

    # ---- Forward request to OpenAI API ----
    openai_api_url = _os.getenv("OPENAI_API_URL", "https://api.openai.com/v1")
    openai_api_key = _os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    if stream:
        # Handle streaming response
        from fastapi.responses import StreamingResponse

        response = forward_request_to(new_body, openai_api_url, openai_api_key, stream=True)

        async def response_generator():
            async for chunk in response.aiter_bytes():
                yield chunk

        return StreamingResponse(
            response_generator(),
            media_type="application/json",
            headers={"Content-Type": "text/event-stream"}
        )
    else:
        # Handle regular response (which may include function calls)
        response = forward_request_to(new_body, openai_api_url+"/chat/completions", openai_api_key)
        return JSONResponse(content=response)


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("front:app", host="0.0.0.0", port=8000, reload=True)
