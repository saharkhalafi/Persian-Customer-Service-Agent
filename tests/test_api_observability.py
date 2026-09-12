from types import SimpleNamespace

from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()

from app.api.dependencies import get_agent
from app.core.context import RequestContext
from app.core.rate_limit import reset_chat_limiter
from app.main import app


class FakeAgentResult:
    def __init__(self, answer="پاسخ آزمایشی", tool_calls=None):
        self.answer = answer
        self.tool_calls = tool_calls or [
            SimpleNamespace(name="search_products", arguments={"query": "کرم"})
        ]


class FakeAgent:
    def __init__(self, result=None, error=None):
        self.result = result or FakeAgentResult()
        self.error = error
        self.calls = []

    def run(self, user_message, context: RequestContext):
        self.calls.append((user_message, context))
        if self.error:
            raise self.error
        return self.result


def _client(agent: FakeAgent) -> TestClient:
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app, raise_server_exceptions=False)


def teardown_function():
    app.dependency_overrides.clear()
    reset_chat_limiter()


def test_docs_and_redoc_are_available():
    client = _client(FakeAgent())
    docs = client.get("/docs")
    redoc = client.get("/redoc")
    openapi = client.get("/openapi.json")
    assert docs.status_code == 200
    assert redoc.status_code == 200
    assert openapi.status_code == 200
    schema = openapi.json()
    assert "/api/v1/chat" in schema["paths"]
    assert "/api/v1/feedback" in schema["paths"]
    chat = schema["paths"]["/api/v1/chat"]["post"]
    assert "ChatRequest" in str(schema)
    assert "ChatResponse" in str(schema)
    assert "401" in chat["responses"]
    assert "500" in chat["responses"]
    assert "X-Customer-ID" in str(schema)


def test_chat_uses_real_agent_override_and_returns_trace_ids():
    agent = FakeAgent()
    client = _client(agent)
    response = client.post(
        "/api/v1/chat",
        json={"message": "کرم ضد چروک پرایم دارید؟"},
        headers={
            "X-Customer-ID": "9206288",
            "X-Request-ID": "req-fixed-1",
            "X-Trace-ID": "trace-fixed-1",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "پاسخ آزمایشی"
    assert body["tool_calls"][0]["name"] == "search_products"
    assert body["request_id"] == "req-fixed-1"
    assert body["trace_id"] == "trace-fixed-1"
    assert response.headers["X-Request-ID"] == "req-fixed-1"
    assert response.headers["X-Trace-ID"] == "trace-fixed-1"
    message, context = agent.calls[0]
    assert message == "کرم ضد چروک پرایم دارید؟"
    assert context.customer_id == "9206288"
    assert context.trace_id == "trace-fixed-1"


def test_missing_customer_header_is_401_envelope():
    client = _client(FakeAgent())
    response = client.post(
        "/api/v1/chat",
        json={"message": "سلام"},
        headers={"X-Trace-ID": "trace-401"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "AUTHENTICATION_ERROR"
    assert "trace_id" in body["error"]
    assert "Traceback" not in str(body)
    assert "detail" not in body


def test_invalid_body_is_400_envelope():
    client = _client(FakeAgent())
    response = client.post(
        "/api/v1/chat",
        json={"message": ""},
        headers={"X-Customer-ID": "9206288"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_agent_timeout_is_503_without_internals():
    client = _client(FakeAgent(error=TimeoutError("deadline exceeded")))
    response = client.post(
        "/api/v1/chat",
        json={"message": "سفارش من چی شد؟"},
        headers={"X-Customer-ID": "9206288", "X-Trace-ID": "trace-503"},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert body["error"]["trace_id"] == "trace-503"
    assert "deadline" not in body["error"]["message"]
    assert "Traceback" not in str(body)


def test_unknown_route_is_404_envelope():
    client = _client(FakeAgent())
    response = client.get("/api/v1/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_health_still_works():
    client = _client(FakeAgent())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "X-Trace-ID" in response.headers


def test_ready_returns_structured_json():
    client = _client(FakeAgent())
    response = client.get("/ready")
    assert response.status_code in {200, 503}
    body = response.json()
    assert "checks" in body
    assert "database" in body["checks"]
    if response.status_code == 200:
        assert body["status"] == "ready"
        assert body["checks"]["database"] == "ok"
    else:
        assert body["status"] == "not_ready"
        assert body["error"]["code"] == "DEPENDENCY_UNAVAILABLE"
