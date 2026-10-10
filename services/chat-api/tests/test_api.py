import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi.testclient import TestClient
from fastmcp.client.client import CallToolResult

import app.main as main

app = main.app
settings = main.settings


def test_health_endpoint():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_empty_message_is_rejected():
    response = TestClient(app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": ""}]})
    assert response.status_code == 422


def test_model_catalog_contains_both_providers_without_secrets():
    response = TestClient(app).get("/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert [provider["id"] for provider in payload["providers"]] == ["openai", "gemini"]
    assert "api_key" not in response.text.lower()


def test_fastmcp_tool_result_is_reduced_to_json_safe_data():
    result = CallToolResult(
        content=[],
        structured_content={"temperature": 72},
        meta=None,
        data={"temperature": 72},
    )

    serialized = main.dump_value(result)

    assert serialized == {"temperature": 72}
    assert json.loads(json.dumps(serialized)) == {"temperature": 72}


def test_unknown_model_is_rejected_before_calling_a_vendor(monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    response = TestClient(app).post("/v1/chat/completions", json={
        "provider": "openai",
        "model": "not-an-enabled-model",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 400
    assert "not enabled" in response.json()["detail"]


def test_unconfigured_gemini_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    response = TestClient(app).post("/v1/chat/completions", json={
        "provider": "gemini",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 503
    assert response.json()["detail"] == "GEMINI_API_KEY is not configured on the server."


def test_selected_provider_and_model_are_used(monkeypatch):
    seen: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            message = SimpleNamespace(tool_calls=None, content="Hello from Gemini")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    @asynccontextmanager
    async def fake_connected_tools():
        yield SimpleNamespace(tools=[])

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(main, "provider_client", lambda provider: fake_client)
    monkeypatch.setattr(main, "connected_tools", fake_connected_tools)

    response = TestClient(app).post("/v1/chat/completions", json={
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


def test_streaming_completion_emits_sse_deltas(monkeypatch):
    seen: dict = {}

    async def chunks():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None))])
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="stream", tool_calls=None))])

    class FakeCompletions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return chunks()

    @asynccontextmanager
    async def fake_connected_tools():
        yield SimpleNamespace(tools=[])

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(main, "provider_client", lambda provider: SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    ))
    monkeypatch.setattr(main, "connected_tools", fake_connected_tools)

    response = TestClient(app).post("/v1/chat/completions", json={
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


def test_streaming_completion_runs_mcp_tool_round(monkeypatch):
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
    async def fake_connected_tools():
        yield FakeTools()

    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    monkeypatch.setattr(main, "provider_client", lambda provider: SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions()),
    ))
    monkeypatch.setattr(main, "connected_tools", fake_connected_tools)

    response = TestClient(app).post("/v1/chat/completions", json={
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
