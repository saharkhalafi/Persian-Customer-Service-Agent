from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from app.core.metrics import record_from_event

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_llm_purpose: ContextVar[str] = ContextVar("llm_purpose", default="")
_TRACEPARENT_RE = re.compile(
    r"^([0-9a-fA-F]{2})-([0-9a-fA-F]{32})-([0-9a-fA-F]{16})-([0-9a-fA-F]{2})$"
)

# Named boundaries for a future OpenTelemetry exporter. The current runtime
# records the same names as structured logs + in-process metrics.
SPAN_BOUNDARIES = (
    "http_request",
    "agent",
    "llm",
    "tool",
    "product_search",
    "database",
)

logger = logging.getLogger("app.observability")

_GEMINI_FLASH_INPUT_USD_PER_M = 0.15
_GEMINI_FLASH_OUTPUT_USD_PER_M = 0.60


def generate_id() -> str:
    return uuid.uuid4().hex


def parse_traceparent(value: str | None) -> str:
    """Extract a W3C trace-id when present. Empty string if unused/invalid."""
    text = str(value or "").strip()
    if not text:
        return ""
    match = _TRACEPARENT_RE.fullmatch(text)
    if not match:
        return ""
    return match.group(2).lower()


def get_span_id() -> str:
    return _span_id.get()


def normalize_incoming_id(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > 128:
        return ""
    if any(ch.isspace() for ch in text):
        return ""
    return text


def resolve_request_ids(
    request_id: str | None,
    trace_id: str | None,
) -> tuple[str, str]:
    incoming_request = normalize_incoming_id(request_id)
    incoming_trace = normalize_incoming_id(trace_id)
    resolved_request = incoming_request or generate_id()
    resolved_trace = incoming_trace or incoming_request or resolved_request
    return resolved_request, resolved_trace


def bind_request_ids(request_id: str, trace_id: str) -> tuple[Any, Any]:
    return _request_id.set(request_id), _trace_id.set(trace_id)


def reset_request_ids(tokens: tuple[Any, Any]) -> None:
    _request_id.reset(tokens[0])
    _trace_id.reset(tokens[1])


def get_request_id() -> str:
    return _request_id.get()


def get_trace_id() -> str:
    return _trace_id.get()


def get_llm_purpose() -> str:
    return _llm_purpose.get()


@contextmanager
def llm_purpose_scope(purpose: str) -> Iterator[None]:
    token = _llm_purpose.set(purpose)
    try:
        yield
    finally:
        _llm_purpose.reset(token)


def raw_query_logging_enabled() -> bool:
    return os.getenv("LOG_RAW_QUERIES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def safe_query_ref(query: str) -> dict[str, Any]:
    text = str(query or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    payload: dict[str, Any] = {
        "query_hash": digest,
        "query_len": len(text),
    }
    if raw_query_logging_enabled() and text:
        payload["query_preview"] = text[:40]
    return payload


def estimate_llm_cost_usd(
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
) -> float | None:
    _ = model
    if input_tokens <= 0 and output_tokens <= 0:
        return None
    cost = (
        (input_tokens / 1_000_000) * _GEMINI_FLASH_INPUT_USD_PER_M
        + (output_tokens / 1_000_000) * _GEMINI_FLASH_OUTPUT_USD_PER_M
    )
    return round(cost, 8)


def log_event(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "trace_id": get_trace_id() or None,
        "request_id": get_request_id() or None,
    }
    span_id = get_span_id()
    if span_id:
        payload["span_id"] = span_id
    for key, value in fields.items():
        if value is not None:
            payload[key] = value
    record_from_event(event, payload)
    logger.info("%s", json.dumps(payload, ensure_ascii=False, default=str))


@contextmanager
def observe(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    started = time.perf_counter()
    extras: dict[str, Any] = {}
    status = "success"
    error_type = None
    if logger.isEnabledFor(logging.DEBUG):
        log_event(f"{event}_started", **fields)
    try:
        yield extras
    except Exception as exc:
        status = "failure"
        error_type = type(exc).__name__
        extras["error_type"] = error_type
        raise
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        log_event(
            event,
            status=status,
            latency_ms=latency_ms,
            **fields,
            **extras,
        )


def traced(event: str, **static_fields: Any):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with observe(event, **static_fields):
                return func(*args, **kwargs)

        return wrapper

    return decorator
