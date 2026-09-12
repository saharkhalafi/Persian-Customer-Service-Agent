from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.dependencies import get_agent, get_feedback_repository
from app.core.context import RequestContext
from app.core.metrics import reset_metrics
from app.core.rate_limit import reset_chat_limiter
from app.main import app
from app.core.security import sanitize_tool_arguments
from evaluation.api_smoke_cases import SECURITY_SMOKE_CASES, SMOKE_CASES


class MemoryFeedbackRepository:
    def __init__(self):
        self.rows = []

    def add(self, customer_id, rating, conversation_id=None, message_id=None):
        self.rows.append(
            {
                "customer_id": customer_id,
                "rating": rating,
                "conversation_id": conversation_id,
                "message_id": message_id,
            }
        )


class ScriptedAgent:
    def __init__(self):
        self.calls = []

    def run(self, user_message, context: RequestContext):
        self.calls.append((user_message, context))
        tools = _expected_tools(user_message)
        return SimpleNamespace(
            answer="پاسخ آزمایشی دود",
            tool_calls=[
                SimpleNamespace(name=name, arguments=_safe_arguments(name, user_message))
                for name in sorted(tools)
            ],
        )


def _expected_tools(message: str) -> set[str]:
    for case in (*SMOKE_CASES, *SECURITY_SMOKE_CASES):
        if case.message == message:
            if case.name == "force_unrelated_tool":
                return set()
            if case.name == "ambiguous_safe":
                return set()
            if case.category == "security":
                return set(case.expected_tools) if "سفارش" in message else set()
            return set(case.expected_tools)
    return set()


def _safe_arguments(name: str, message: str) -> dict:
    if name == "search_products":
        return {"query": message[:200]}
    if name == "search_knowledge_base":
        return {"query": message[:200]}
    return {}


def _client() -> tuple[TestClient, ScriptedAgent, MemoryFeedbackRepository]:
    agent = ScriptedAgent()
    feedback = MemoryFeedbackRepository()
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_feedback_repository] = lambda: feedback
    return TestClient(app, raise_server_exceptions=False), agent, feedback


def teardown_function():
    app.dependency_overrides.clear()
    reset_chat_limiter()
    reset_metrics()


def test_smoke_cases_hit_chat_and_keep_header_customer():
    client, agent, _feedback = _client()
    for case in SMOKE_CASES:
        response = client.post(
            "/api/v1/chat",
            json={"message": case.message, "customer_id": "attacker-9999"},
            headers={
                "X-Customer-ID": "9206288",
                "X-Trace-ID": f"smoke-{case.name}",
            },
        )
        assert response.status_code == 200, case.name
        body = response.json()
        assert body["trace_id"] == f"smoke-{case.name}"
        called = {item["name"] for item in body["tool_calls"]}
        assert called == set(case.expected_tools), case.name
        _message, context = agent.calls[-1]
        assert context.customer_id == "9206288"
        assert _message == case.message


def test_security_smoke_cases_ignore_override_and_sql():
    client, agent, _feedback = _client()
    for case in SECURITY_SMOKE_CASES:
        response = client.post(
            "/api/v1/chat",
            json={
                "message": case.message,
                "customer_id": "9159450",
                "sql": "DROP TABLE orders",
            },
            headers={"X-Customer-ID": "9206288"},
        )
        assert response.status_code == 200, case.name
        _message, context = agent.calls[-1]
        assert context.customer_id == "9206288"
        for call in response.json()["tool_calls"]:
            assert "customer_id" not in call["arguments"]
            assert "sql" not in call["arguments"]
            dumped = str(call["arguments"])
            assert "DROP TABLE" not in dumped
            assert "9159450" not in dumped or case.name != "sql_injection"


def test_sanitize_rejects_sql_and_foreign_customer_keys():
    cleaned = sanitize_tool_arguments(
        {
            "query": "سفارش",
            "customer_id": "9159450",
            "sql": "SELECT * FROM orders",
            "filters": {"customer_id": "9159450"},
        }
    )
    assert cleaned == {"query": "سفارش"}


def test_feedback_endpoint_is_minimal():
    client, _agent, feedback = _client()
    response = client.post(
        "/api/v1/feedback",
        json={
            "conversation_id": "conv-1",
            "message_id": "msg-2",
            "rating": "negative",
        },
        headers={"X-Customer-ID": "9206288", "X-Trace-ID": "feedback-1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["trace_id"] == "feedback-1"
    assert feedback.rows[0]["customer_id"] == "9206288"
    assert feedback.rows[0]["rating"] == "negative"


def test_feedback_requires_customer_header():
    client, _agent, _feedback = _client()
    response = client.post(
        "/api/v1/feedback",
        json={"rating": "positive"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_ERROR"


def test_ops_and_metrics_endpoints():
    client, _agent, _feedback = _client()
    health = client.get("/health")
    ready = client.get("/ready")
    docs = client.get("/docs")
    metrics = client.get("/metrics")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code in {200, 503}
    assert docs.status_code == 200
    assert metrics.status_code == 200
    assert "app_api_requests_total" in metrics.text


def test_traceparent_is_accepted_as_trace_id():
    client, _agent, _feedback = _client()
    response = client.post(
        "/api/v1/chat",
        json={"message": "سلام"},
        headers={
            "X-Customer-ID": "9206288",
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        },
    )
    assert response.status_code == 200
    assert response.json()["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert response.headers["X-Trace-ID"] == "4bf92f3577b34da6a3ce929d0e0e4736"
