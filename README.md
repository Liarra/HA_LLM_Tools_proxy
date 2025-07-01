# OpenAI API Proxy with Tool Optimization

**WARNING: This software is very much at the "proof of concept" stage and misses a lot of essential features. 
It's not anywhere near production ready. Use at your own peril. MRs and suggestions are welcome.**

This is a proxy server that sits between your HomeAssistant LLM integration and the OpenAI-compatible API and automatically 
debloats your requests to the OpenAI-compatible LLM.

The biggest source of prompt bloat in HomeAssistant LLM calls is the inclusion of all available tools in every request.
This proxy server intercepts requests to the `/v1/chat/completions` endpoint, analyzes the last user message, and 
selects only the most relevant tools to include with the request using an embedding model.
Then the request is forwarded to the actual LLM API.

This provides faster response times, helps you to stay within your token budget, and reduces the energy burden 
on the environment.

## Setup

### Docker
This is probably the easiest way to run this proxy server.

1. **Build the Docker image**:
   ```bash
   docker build -t openai-proxy .
   ```
2. **Create a `.env` file** in the same directory as your Dockerfile with the following content:
   ```bash
    OPENAI_API_KEY=your-api-key
    OPENAI_API_URL=your-llm-endpoint/v1
    TOOLS_TO_KEEP=3 # Or however many tools you want to keep in the request
    ```
3. **Run the Docker container**:
   ```bash
    docker run -d -p 8000:8000 -v ./data:data --env-file .env openai-proxy
    ```
   
### Python Script
If you prefer to run the proxy server as a Python script, just create the `.env` file as described above, install the 
required dependencies from `requirements.txt`, and run the script:

```bash
python front.py
```

## Usage
Once the server is running, set it up as the endpoint for your HomeAssistant LLM integration.
I use the amazing [OpenAI Compatible Conversation](https://github.com/michelle-avery/openai-compatible-conversation) 
and its fork [No-think LLM](https://github.com/duckida/ha-nothink-llm) optimised for Qwen and DeepSeek output.

Set up your integration as usual, setting up the system prompt and all the model parameters. 
The only difference is that you will set the API endpoint to the URL of this proxy server, 
for example. `http://localhost:8000/v1/, or wherever you are running the proxy server.

The proxy will receive the requests to the `/v1/chat/completions` endpoint and pass them unchanged,
except for only selecting the subset of the tools. It will then pass the response from the LLM back to HomeAssistant 
unchanged.

## Current Limitations

By definition, this concept introduces some restrictions on how creative your prompt can be. You can no longer say
"_Make me regret my life_" and expect the LLM to play Despacito, because the proxy is not guaranteed to include the
media control tool with such a request. The LLM might understand you, but the embedding model in the proxy will not 
provide it with a tool to actually fulfil your request. In practice this means being more explicit with your prompts. 
("_Turn on the lights_" instead of "_Hello darkness my old friend_", "_Play some music_" instead of 
"_Make me feel better_".)

There is a primitive fallback mechanism for very short prompts (designed to catch the use case of an assistant asking 
for confirmation before executing a tool).

That being said, there are also some technical limitations:

- No support for changing the list of tools. If you expose new tools, the proxy will pick the new additions up, but 
deleting previously known tools is not supported yet. To avoid model picking up tools that are no longer available, 
after changing the list of tools on HomeAssistant side, one should:
    1. Stop the proxy server
    2. Delete the files in `data` folder
    3. Restart the proxy server
    4. Accept the delay at the fisrt request as the model re-embeds the tool definitions.

- No multiprocessing (or even multithreading, really) support. This is probably fine for light home use for now. 
Just don't share this proxy server between multiple HomeAssistant instances. 
- Streaming responses is not really well tested.
- No smart context support. The proxy only looks at the last user message (and the previous assistant message, if the 
last user message is very short) to decide which tools to include.

That all being said, the proxy works well for my use case and I hope it will work well for you too.