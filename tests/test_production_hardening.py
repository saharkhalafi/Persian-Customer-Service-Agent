from app.core.config import GEMINI_RETRY_ATTEMPTS, GEMINI_TIMEOUT_MS
from app.core.database import ENGINE_CONNECT_ARGS, engine
from app.core.rate_limit import get_chat_limiter, reset_chat_limiter
from app.core.security import redact_secrets, validate_customer_id
from app.core.exceptions import AuthenticationError
from app.api.dependencies import get_agent
from app.main import app
from fastapi.testclient import TestClient
from tests.test_api_observability import FakeAgent


def teardown_function():
    app.dependency_overrides.clear()
    reset_chat_limiter()


def _client(agent: FakeAgent) -> TestClient:
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app, raise_server_exceptions=False)


def test_gemini_timeout_and_retries_are_bounded():
    assert GEMINI_TIMEOUT_MS <= 120_000
    assert GEMINI_RETRY_ATTEMPTS <= 3
    assert GEMINI_RETRY_ATTEMPTS >= 1


def test_database_engine_has_pool_and_timeouts():
    assert engine.pool.size() >= 1
    assert getattr(engine.pool, "_timeout", 1) >= 1
    if str(engine.url).startswith("postgres"):
        assert "connect_timeout" in ENGINE_CONNECT_ARGS
        assert "statement_timeout" in str(ENGINE_CONNECT_ARGS.get("options", ""))


def test_invalid_customer_id_is_401():
    client = _client(FakeAgent())
    response = client.post(
        "/api/v1/chat",
        json={"message": "سلام"},
        headers={"X-Customer-ID": "1; DROP TABLE users"},
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "AUTHENTICATION_ERROR"
    assert "DROP TABLE" not in str(body)


def test_oversized_chat_body_is_rejected():
    from app.core.exceptions import ValidationFailedError
    from app.core.security import enforce_content_length

    try:
        enforce_content_length("999999", 16_384)
        assert False
    except ValidationFailedError:
        pass


def test_chat_rate_limit_returns_429_envelope():
    limiter = get_chat_limiter()
    limiter.reset()
    original = limiter.max_requests
    limiter.max_requests = 2
    client = _client(FakeAgent())
    try:
        headers = {"X-Customer-ID": "9206288"}
        first = client.post("/api/v1/chat", json={"message": "سلام"}, headers=headers)
        second = client.post("/api/v1/chat", json={"message": "سلام"}, headers=headers)
        third = client.post("/api/v1/chat", json={"message": "سلام"}, headers=headers)
        assert first.status_code == 200
        assert second.status_code == 200
        assert third.status_code == 429
        body = third.json()
        assert body["error"]["code"] == "RATE_LIMIT"
        assert "trace_id" in body["error"]
        assert third.headers.get("Retry-After") == "60"
    finally:
        limiter.max_requests = original
        limiter.reset()


def test_validate_customer_id_accepts_existing_ids():
    assert validate_customer_id("9206288") == "9206288"


def test_validate_customer_id_rejects_empty():
    try:
        validate_customer_id("  ")
        assert False
    except AuthenticationError:
        pass


def test_redact_secrets_from_log_text():
    text = redact_secrets("GEMINI_API_KEY=abc123 DATABASE_URL=postgres://x")
    assert "abc123" not in text
    assert "[REDACTED]" in text
