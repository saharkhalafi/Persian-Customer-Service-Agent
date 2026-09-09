from app.services.order_service import OrderService
from app.schemas.order_schemas import CustomerOrderSummary


class OrderTools:

    def __init__(self, order_service: OrderService):
        self.order_service = order_service

    def get_customer_order_summary(
        self,
        customer_id: str,
    ) -> CustomerOrderSummary:

        return self.order_service.get_customer_order_summary(
            customer_id=customer_id
        )