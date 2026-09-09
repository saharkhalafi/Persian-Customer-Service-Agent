from app.core.context import RequestContext
from app.services.customer_service import CustomerService


class CustomerTools:
    def __init__(
        self,
        service: CustomerService,
    ):
        self.service = service

    def get_customer_profile(
        self,
        context: RequestContext,
        fields: list[str] | None = None,
    ):
        return self.service.get_customer_profile(
            customer_id=context.customer_id,
            fields=fields,
        )