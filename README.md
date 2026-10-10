# MCP Chat Tutorial

A runnable tutorial for a ChatGPT-like web application. A React client calls a Python
chat API. The chat API calls an OpenAI model and two independent Model Context Protocol
(MCP) services for live information.

## Architecture

```text
React browser (3000) -> FastAPI chat API (8000) -> OpenAI Responses API
                              |                 
                     streamable HTTP MCP
                         |             |
                  Search MCP (8001)  Live-data MCP (8002)
                  web search          weather / sports / stocks
```

The folders in `services/` are standalone Python microservices, each with its own
Dockerfile and requirements. `apps/web` is an independent React/Vite frontend.

## Run locally

1. Create an API Platform key at <https://platform.openai.com/api-keys>. A ChatGPT
   subscription is not an API credential; this app requires an API key from the same
   OpenAI account and never sends it to the browser.
2. Run `cp .env.example .env`, then set `OPENAI_API_KEY` in `.env`.
3. Run `docker compose up --build`.
4. Open <http://localhost:3000>. The API’s interactive documentation is at
   <http://localhost:8000/docs>.

Try: “What is the weather in Chicago?”, “AAPL stock price”, “NBA scores today”, or
“What happened in AI today?”. The UI shows the tool calls that supplied live data.

Use `docker compose down` to stop the stack. The public tutorial data sources need no
key but can apply limits; production apps should use licensed providers and caching.

## API example

`POST /v1/chat/completions`

```json
{"messages": [{"role": "user", "content": "Weather in Austin?"}]}
```

The endpoint lists tools from each MCP service, presents them as OpenAI function tools,
executes selected calls, sends the outputs back to the model, then returns its answer.

## Build individual services

```bash
docker build -t tutorial-search ./services/mcp-search
docker run --rm -p 8001:8001 tutorial-search

docker build -t tutorial-web ./apps/web
docker run --rm -p 3000:80 tutorial-web
```

## Security checklist

- Keep `OPENAI_API_KEY` in `.env` or a secret manager, never in React code or Git.
- Add authentication, rate limits, audit logs, and request limits before production.
- Search snippets are untrusted reference data, never instructions for the model.

The chat service follows the [official OpenAI quickstart](https://developers.openai.com/api/docs/quickstart): the Python SDK reads an API key from the server environment and uses the Responses API for tool calling.
