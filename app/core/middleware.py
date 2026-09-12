from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import CHAT_MAX_BODY_BYTES
from app.core.errors import _json_error
from app.core.exceptions import ValidationFailedError
from app.core.observability import (
    bind_request_ids,
    log_event,
    parse_traceparent,
    reset_request_ids,
    resolve_request_ids,
)
from app.core.security import enforce_content_length


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_trace = request.headers.get("x-trace-id") or parse_traceparent(
            request.headers.get("traceparent")
        )
        request_id, trace_id = resolve_request_ids(
            request.headers.get("x-request-id"),
            incoming_trace,
        )
        request.state.request_id = request_id
        request.state.trace_id = trace_id
        tokens = bind_request_ids(request_id, trace_id)
        started = time.perf_counter()
        status_code = 500
        try:
            if request.url.path == "/api/v1/chat" and request.method == "POST":
                try:
                    enforce_content_length(
                        request.headers.get("content-length"),
                        CHAT_MAX_BODY_BYTES,
                    )
                except ValidationFailedError:
                    status_code = 400
                    response = _json_error(400, "VALIDATION_ERROR", request=request)
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Trace-ID"] = trace_id
                    return response
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            return response
        finally:
            log_event(
                "request_completed",
                endpoint=request.url.path,
                method=request.method,
                status=status_code,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
            reset_request_ids(tokens)
