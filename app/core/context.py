from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    customer_id: str
    request_id: str = ""
    trace_id: str = ""
