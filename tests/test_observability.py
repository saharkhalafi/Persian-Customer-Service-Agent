import json
import logging

from app.core.context import RequestContext
from app.core.observability import (
    bind_request_ids,
    estimate_llm_cost_usd,
    get_trace_id,
    log_event,
    reset_request_ids,
    resolve_request_ids,
    safe_query_ref,
)


def test_request_context_still_accepts_customer_id_only():
    context = RequestContext(customer_id="9206288")
    assert context.customer_id == "9206288"
    assert context.request_id == ""
    assert context.trace_id == ""


def test_resolve_ids_generates_and_propagates():
    request_id, trace_id = resolve_request_ids(None, None)
    assert request_id
    assert trace_id == request_id

    request_id, trace_id = resolve_request_ids("abc123", None)
    assert request_id == "abc123"
    assert trace_id == "abc123"

    request_id, trace_id = resolve_request_ids("req-1", "trace-9")
    assert request_id == "req-1"
    assert trace_id == "trace-9"


def test_rejects_unsafe_incoming_ids():
    request_id, _trace_id = resolve_request_ids("bad id with spaces", "x" * 200)
    assert request_id != "bad id with spaces"
    assert len(request_id) == 32


def test_safe_query_ref_redacts_by_default(monkeypatch):
    monkeypatch.delenv("LOG_RAW_QUERIES", raising=False)
    payload = safe_query_ref("کرم ضد چروک پرایم دارید؟")
    assert "query_hash" in payload
    assert "query" not in payload
    assert "query_preview" not in payload


def test_bind_request_ids_visible_to_logs(caplog):
    tokens = bind_request_ids("req-obs", "trace-obs")
    try:
        assert get_trace_id() == "trace-obs"
        with caplog.at_level(logging.INFO, logger="app.observability"):
            log_event("agent_tool_selection", tool="search_products")
        records = [json.loads(record.getMessage()) for record in caplog.records]
        assert records[-1]["event"] == "agent_tool_selection"
        assert records[-1]["trace_id"] == "trace-obs"
        assert records[-1]["tool"] == "search_products"
    finally:
        reset_request_ids(tokens)


def test_estimate_cost_none_without_tokens():
    assert estimate_llm_cost_usd(0, 0) is None
    assert estimate_llm_cost_usd(1_000_000, 0) == 0.15
