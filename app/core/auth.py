from fastapi import Header, HTTPException

from app.core.context import RequestContext


def get_request_context(
    x_customer_id: str | None = Header(default=None),
) -> RequestContext:

    if not x_customer_id:
        raise HTTPException(
            status_code=401,
            detail="Customer authentication is required",
        )

    return RequestContext(
        customer_id=x_customer_id
    )