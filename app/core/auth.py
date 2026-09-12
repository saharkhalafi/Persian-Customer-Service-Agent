from fastapi import Header, Request

from app.core.context import RequestContext
from app.core.observability import bind_request_ids, generate_id
from app.core.security import validate_customer_id


def get_request_context(
    request: Request,
    x_customer_id: str | None = Header(
        default=None,
        alias="X-Customer-ID",
        description="Authenticated customer identifier supplied by the backend/session.",
    ),
) -> RequestContext:
    request_id = str(getattr(request.state, "request_id", "") or "") or generate_id()
    trace_id = str(getattr(request.state, "trace_id", "") or "") or request_id
    bind_request_ids(request_id, trace_id)

    return RequestContext(
        customer_id=validate_customer_id(x_customer_id),
        request_id=request_id,
        trace_id=trace_id,
    )
