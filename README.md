# Home Assistant LLM Tools Proxy

An OpenAI-compatible reverse proxy that reduces Home Assistant tool-schema
prompt bloat. It embeds the current user request and the tools supplied with
that request, then forwards only the relevant subset to the upstream LLM.

The proxy supports both ordinary and streamed Chat Completions responses. Other
`/v1/*` routes, including `/v1/models`, are passed through transparently so
clients such as Custom Conversation can treat it as the model server.

> This remains a small self-hosted project. Keep it on a trusted network and do
> not expose it directly to the internet.

## Docker setup

Build the image:

```bash
docker build -t ha-llm-tools-proxy .
```

Create `.env`:

```dotenv
OPENAI_API_KEY=your-upstream-key
OPENAI_API_URL=http://your-vllm-host:8000/v1

# Hard ceiling, including whitelisted tools.
TOOLS_TO_KEEP=3

# Relevant tools below this cosine-similarity score are omitted. This means the
# proxy may return fewer than TOOLS_TO_KEEP tools. E5 scores are model-specific;
# 0.75 is a reasonable starting point, not a universal truth.
MIN_TOOL_SIMILARITY=0.75

# Comma-separated tool names.
WHITELISTED_TOOLS=GetLiveContext
BLACKLISTED_TOOLS=HassHumidifierMode,HassHumidifierSetPoint
```

Run it:

```bash
docker run -d \
  --name ha-llm-tools-proxy \
  -p 8000:8000 \
  -v ./data:/app/data \
  --env-file .env \
  ha-llm-tools-proxy
```

Point Home Assistant or Custom Conversation at
`http://proxy-host:8000/v1`. The proxy forwards from there to
`OPENAI_API_URL`.

## How selection works

For each Chat Completions request, the proxy:

1. Considers only tools included in that request. Deleted HA tools cannot leak
   back in from old state.
2. Always prioritizes a specifically named `tool_choice`, then configured
   whitelist entries.
3. Excludes blacklisted tools unless a specifically named `tool_choice`
   requires one.
4. Scores the other tools against the latest user message using E5 retrieval
   prefixes and each tool's name, description, and parameter schema.
5. Keeps scores at or above `MIN_TOOL_SIMILARITY` until `TOOLS_TO_KEEP` is
   reached.

`TOOLS_TO_KEEP` is a hard maximum. Whitelisted tools consume slots. If the
threshold accepts only one semantic match and one whitelist tool is present,
the proxy returns two tools even when the maximum is three.

With `tool_choice: "required"`, at least one candidate is retained when
possible, even if every score is below the threshold. With a named tool choice,
that exact tool receives first priority.

Tool embeddings are cached in `data/tool_embeddings.sqlite3`. The cache key
includes the complete canonical tool schema and embedding-model name, so schema
changes generate fresh embeddings. The embedding model itself is loaded lazily
on the first request that needs semantic selection.

## Configuration

| Variable | Default | Meaning |
|---|---:|---|
| `OPENAI_API_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible `/v1` base URL |
| `OPENAI_API_KEY` | empty | Authorization key sent upstream; incoming authorization is preserved when empty |
| `TOOLS_TO_KEEP` | `3` | Hard maximum number of tools forwarded |
| `MIN_TOOL_SIMILARITY` | `0.75` | Minimum cosine similarity for non-mandatory tools |
| `WHITELISTED_TOOLS` | `GetLiveContext` | Tools given priority on every request |
| `BLACKLISTED_TOOLS` | `HassHumidifierMode,HassHumidifierSetPoint` | Tools normally excluded |
| `EMBEDDING_MODEL` | `intfloat/e5-small-v2` | Hugging Face embedding model |
| `EMBEDDING_CACHE` | `data/tool_embeddings.sqlite3` | SQLite cache path |
| `LOG_LEVEL` | `INFO` | Python log level |

Set either comma-separated tool list explicitly to an empty value to disable
that list, for example `WHITELISTED_TOOLS=`.

To tune the threshold, run representative HA phrases at `LOG_LEVEL=DEBUG` and
inspect the selected semantic scores. Raise the threshold to reduce prompt
size; lower it if legitimate tools are being missed.

## Local development and tests

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
pytest --ignore=tests/test_litellm_compat.py
```

The compatibility test drives the proxy through LiteLLM's async `Router` with
`stream=True` and `stream_options={"include_usage": True}`, matching Custom
Conversation's current request path:

```bash
python -m pip install -r requirements-compat.txt
pytest -m compatibility
```

CI runs that test against the pinned LiteLLM version from Custom Conversation's
current manifest. A weekly/manual job reads the latest Custom Conversation
manifest and tests whichever LiteLLM version it then requires.

## Known trade-off

Semantic filtering deliberately constrains indirect requests. “Turn on the
kitchen lights” is safer than expecting an embedding model to infer the light
tool from “hello darkness, my old friend.” The proxy fails open—forwarding the
original tool list—if the selector itself errors, so an embedding failure does
not take Home Assistant offline.
