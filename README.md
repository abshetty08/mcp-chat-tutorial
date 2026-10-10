# MCP Chat Tutorial

A runnable tutorial for a multi-provider chat application. A React client calls a
Python chat API, where the user-selected OpenAI or Google Gemini model can call two
independent Model Context Protocol (MCP) services for live information.

## Architecture

```text
React browser (3000) -> FastAPI chat API (8000) -> OpenAI Chat Completions
          |                   |                  -> Gemini OpenAI-compatible API
 provider/model picker + SSE  |
                     streamable HTTP MCP
                         |             |
                  Search MCP (8001)  Live-data MCP (8002)
                  web search          weather / sports / stocks
```

The folders in `services/` are standalone Python microservices, each with its own
Dockerfile and requirements. `apps/web` is an independent React/Vite frontend.

## Code organization

The tutorial uses small objects with explicit responsibilities and keeps framework
entry points as composition roots:

- `services/chat-api/app/providers.py` owns vendor configuration, model allowlists,
  and SDK client creation.
- `services/chat-api/app/mcp_gateway.py` manages MCP connections, discovery, and
  tool invocation.
- `services/chat-api/app/chat_service.py` orchestrates completion and tool-call
  loops without depending on FastAPI routes.
- `services/chat-api/app/streaming.py` encapsulates SSE encoding and incremental
  tool-call reconstruction.
- Each MCP server delegates transport wrappers to domain service classes in
  `search_service.py` or `live_data_services.py`.
- The React component delegates HTTP and SSE state machines to `ChatApiClient` and
  `ServerSentEventStream`; rendering remains functional and hook-based.

Constructor injection keeps provider clients and tool gateways replaceable in tests,
while Pydantic models remain the single API contract shared by the route and service
layers.

## Run locally

1. Create an OpenAI API Platform key at <https://platform.openai.com/api-keys>, a
   Gemini key at <https://aistudio.google.com/app/apikey>, or both. Chat product
   subscriptions are not API credentials. Keys remain in the server container.
2. Run `cp .env.example .env`, then set `OPENAI_API_KEY`, `GEMINI_API_KEY`, or both
   in `.env`. A provider without a key appears disabled in the client.
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
{
  "provider": "gemini",
  "model": "gemini-3.8-flash",
  "stream": true,
  "messages": [{"role": "user", "content": "Weather in Austin?"}]
}
```

Set `stream` to `true` to receive `text/event-stream` events. The gateway emits
`delta` events for incremental text, `tool_call` events after MCP tools complete,
`done` when the response finishes, and `error` if a failure occurs after streaming
has started. Omit `stream` or set it to `false` for the original JSON response.
For Gemini thinking models, the gateway also preserves required
[thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures)
between streamed function-call rounds without exposing them to the browser.

`GET /v1/models` returns the server-controlled provider/model catalog and whether each
provider is configured. The chat endpoint validates the requested pair, lists tools
from each MCP service, presents them as function tools, executes selected calls, sends
the outputs back to the chosen vendor, then returns or streams its answer.

Model choices can be changed without code by editing the comma-separated
`OPENAI_MODELS` and `GEMINI_MODELS` environment variables. The allow lists prevent a
browser from requesting arbitrary or unexpectedly expensive models.

## Build individual services

```bash
docker build -t tutorial-search ./services/mcp-search
docker run --rm -p 8001:8001 tutorial-search

docker build -t tutorial-web ./apps/web
docker run --rm -p 3000:80 tutorial-web
```

## Security checklist

- Keep `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env` or a secret manager, never in
  React code, `.env.example`, or Git.
- Add authentication, rate limits, audit logs, and request limits before production.
- Search snippets are untrusted reference data, never instructions for the model.

The gateway uses the OpenAI Python client for both vendors. OpenAI supports function
tools through [Chat Completions](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create),
and Google documents the same client shape through its [OpenAI compatibility endpoint](https://ai.google.dev/gemini-api/docs/openai).
