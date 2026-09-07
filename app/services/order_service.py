from app.repositories.order_repository import OrderRepository


class OrderService:

    def __init__(self, order_repository: OrderRepository):
        self.order_repository = order_repository

    def get_customer_order_summary(self, customer_id: str) -> dict:
        summary = self.order_repository.get_order_summary(
            customer_id
        )

        return {
            "customer_id": customer_id,
            "total_orders": summary["total_orders"],
            "successful_orders": summary["successful_orders"],
            "cancelled_orders": summary["cancelled_orders"],
            "total_items": summary["total_items"],
            "total_spent": float(summary["total_spent"] or 0),
        }