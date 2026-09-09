from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    customer_id: str