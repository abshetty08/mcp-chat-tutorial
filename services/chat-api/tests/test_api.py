import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient
from fastmcp.client.client import CallToolResult

from app.chat_service import ChatCompletionService
from app.config import Settings
from app.main import create_app
from app.openai_compat import OpenAICompatibilityAdapter
from app.providers import ProviderRegistry
from app.serialization import JsonValueSerializer


@asynccontextmanager
async def empty_tools():
    yield SimpleNamespace(tools=[])


def build_client(
    *,
    settings: Settings | None = None,
    provider_client=None,
    tool_gateway_factory=empty_tools,
) -> tuple[TestClient, ProviderRegistry]:
    settings = settings or Settings(
        _env_file=None,
        openai_api_key="test-openai-key",
        gemini_api_key="test-gemini-key",
    )
    providers = ProviderRegistry(settings)
    if provider_client is not None:
        providers.client_for = lambda _provider: provider_client
    chat_service = ChatCompletionService(providers, tool_gateway_factory)
    container = SimpleNamespace(
        providers=providers,
        chat_service=chat_service,
        openai_compat=OpenAICompatibilityAdapter(providers, chat_service),
    )
    return TestClient(create_app(container)), providers


def test_health_endpoint():
    client, _ = build_client()
    assert client.get("/health").json() == {"status": "ok"}


def test_empty_message_is_rejected():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": ""}]})
    assert response.status_code == 422


def test_model_catalog_contains_both_providers_without_secrets():
    client, _ = build_client()
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert [provider["id"] for provider in response.json()["providers"]] == ["openai", "gemini"]
    assert "api_key" not in response.text.lower()


def test_librechat_model_catalog_uses_openai_shape():
    client, _ = build_client()
    response = client.get("/v1/compat/gemini/models")
    assert response.status_code == 200
    assert response.json()["object"] == "list"
    assert response.json()["data"][0] == {
        "id": "gemini-3.8-flash",
        "object": "model",
        "created": 0,
        "owned_by": "gemini",
    }


def test_unknown_librechat_provider_is_rejected():
    client, _ = build_client()
    response = client.get("/v1/compat/unknown/models")
    assert response.status_code == 422


def test_fastmcp_tool_result_is_reduced_to_json_safe_data():
    result = CallToolResult(
        content=[],
        structured_content={"temperature": 72},
        meta=None,
        data={"temperature": 72},
    )
    serialized = JsonValueSerializer().encode(result)
    assert serialized == {"temperature": 72}
    assert json.loads(json.dumps(serialized)) == {"temperature": 72}


def test_unknown_model_is_rejected_before_calling_a_vendor():
    client, _ = build_client()
    response = client.post("/v1/chat/completions", json={
        "provider": "openai",
        "model": "not-an-enabled-model",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]


def test_unconfigured_gemini_provider_is_rejected():
    client, _ = build_client(settings=Settings(
        _env_file=None,
        openai_api_key="test-openai-key",
        gemini_api_key=None,
    ))
    response = client.post("/v1/chat/completions", json={
        "provider": "gemini",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 503
    assert response.json()["detail"] == "GEMINI_API_KEY is not configured on the server."


def test_selected_provider_and_model_are_used():
    seen: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            message = SimpleNamespace(tool_calls=None, content="Hello from Gemini")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client, _ = build_client(provider_client=fake_client)
    response = client.post("/v1/chat/completions", json={
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert response.json()["model"] == "gemini-3.8-flash"
    assert response.json()["answer"] == "Hello from Gemini"
    assert seen["model"] == "gemini-3.8-flash"
    assert seen["messages"][-1] == {"role": "user", "content": "hello"}


def test_librechat_non_streaming_request_uses_selected_vendor():
    seen: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            message = SimpleNamespace(tool_calls=None, content="LibreChat response")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client, _ = build_client(provider_client=fake_client)
    response = client.post("/v1/compat/openai/chat/completions", json={
        "model": "gpt-6-luna",
        "messages": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "hello"},
        ],
    })
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert response.json()["choices"][0]["message"]["content"] == "LibreChat response"
    assert seen["model"] == "gpt-6-luna"
    assert seen["messages"][-2] == {"role": "system", "content": "Be concise."}


def test_librechat_streaming_request_uses_openai_sse_shape():
    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Hello from LibreChat", tool_calls=None),
        )])

    class FakeCompletions:
        async def create(self, **_kwargs):
            return chunks()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client, _ = build_client(provider_client=fake_client)
    response = client.post("/v1/compat/gemini/chat/completions", json={
        "model": "gemini-3.8-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object": "chat.completion.chunk"' in response.text
    assert '"content": "Hello from LibreChat"' in response.text
    assert '"finish_reason": "stop"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_streaming_completion_emits_sse_deltas():
    seen: dict = {}

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="stream", tool_calls=None))])

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return chunks()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client, _ = build_client(provider_client=fake_client)
    response = client.post("/v1/chat/completions", json={
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: delta\ndata: {"content": "Hello "}' in response.text
    assert 'event: delta\ndata: {"content": "stream"}' in response.text
    assert "event: done" in response.text
    assert seen["stream"] is True


def test_streaming_completion_runs_mcp_tool_round():
    requests: list[dict] = []
    tool_requests: list[tuple[str, dict]] = []

    async def tool_chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0,
                id="call_weather",
                function=SimpleNamespace(name="weather_lookup", arguments='{"location":"Chi'),
                extra_content={"google": {"thought_signature": "opaque-signature"}},
            )],
        ))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments='cago"}'),
                extra_content=None,
            )],
        ))])

    async def answer_chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content="It is sunny.",
            tool_calls=None,
        ))])

    class FakeCompletions:
        async def create(self, **kwargs):
            requests.append(kwargs)
            return tool_chunks() if len(requests) == 1 else answer_chunks()

    class FakeTools:
        tools = [{"type": "function", "function": {"name": "weather_lookup"}}]

        async def call(self, name, arguments):
            tool_requests.append((name, arguments))
            return {"temperature": 72}

    @asynccontextmanager
    async def fake_tools():
        yield FakeTools()

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    client, _ = build_client(provider_client=fake_client, tool_gateway_factory=fake_tools)
    response = client.post("/v1/chat/completions", json={
        "provider": "gemini",
        "model": "gemini-3.8-flash",
        "stream": True,
        "messages": [{"role": "user", "content": "weather?"}],
    })
    assert response.status_code == 200
    assert "event: tool_call" in response.text
    assert '"name": "weather_lookup"' in response.text
    assert 'event: delta\ndata: {"content": "It is sunny."}' in response.text
    assert "opaque-signature" not in response.text
    assert tool_requests == [("weather_lookup", {"location": "Chicago"})]
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert requests[1]["messages"][-2]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "opaque-signature"},
    }
