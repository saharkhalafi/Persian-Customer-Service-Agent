from app.core.metrics import record_from_event, registry, reset_metrics
from app.core.observability import parse_traceparent


def setup_function():
    reset_metrics()


def test_parse_traceparent_extracts_trace_id():
    assert parse_traceparent(None) == ""
    assert parse_traceparent("not-a-trace") == ""
    parsed = parse_traceparent(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    )
    assert parsed == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_record_api_and_llm_metrics():
    record_from_event(
        "request_completed",
        {"endpoint": "/api/v1/chat", "method": "POST", "status": 200, "latency_ms": 12},
    )
    record_from_event(
        "request_completed",
        {"endpoint": "/api/v1/chat", "method": "POST", "status": 500, "latency_ms": 9},
    )
    record_from_event(
        "llm_call",
        {
            "purpose": "agent_reasoning",
            "status": "failure",
            "timeout": True,
            "latency_ms": 40,
            "input_tokens": 10,
            "output_tokens": 2,
            "estimated_cost_usd": 0.001,
        },
    )
    text = registry.render_prometheus()
    assert "app_api_requests_total" in text
    assert "app_api_errors_total" in text
    assert "app_llm_timeouts_total" in text
    assert "app_llm_tokens_total" in text
    assert "app_llm_estimated_cost_usd_total" in text


def test_product_search_and_agent_metrics():
    record_from_event(
        "product_search_completed",
        {
            "empty": True,
            "latency_ms": 80,
            "llm_fallback_triggered": True,
            "llm_failure": True,
        },
    )
    record_from_event("agent_completed", {"latency_ms": 100})
    record_from_event("tool_execution", {"tool": "search_products", "status": "success"})
    record_from_event("agent_tool_selection_error", {"tool": "drop_table"})
    text = registry.render_prometheus()
    assert "app_product_search_empty_total" in text
    assert "app_product_search_llm_fallback_failures_total" in text
    assert "app_agent_tool_selection_errors_total" in text
