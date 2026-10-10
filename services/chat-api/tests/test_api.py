from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_empty_message_is_rejected():
    response = TestClient(app).post("/v1/chat/completions", json={"messages": [{"role": "user", "content": ""}]})
    assert response.status_code == 422
