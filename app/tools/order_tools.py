from app.core.context import RequestContext
from app.schemas.order_schemas import (
    CustomerOrderSearchResult,
    CustomerOrderSummary,
    LatestOrder,
    OrderHistoryItem,
    OrderItem,
    OrderStatus,
    ProductPurchaseHistory,
    PurchasedProduct,
)
from app.services.order_service import OrderService


class OrderTools:

    def __init__(
        self,
        order_service: OrderService,
    ):
        self.order_service = order_service

    def get_customer_order_summary(
        self,
        context: RequestContext,
    ) -> CustomerOrderSummary:

        return self.order_service.get_customer_order_summary(
            customer_id=context.customer_id
        )

    def get_latest_order(
        self,
        context: RequestContext,
    ) -> LatestOrder | None:

        return self.order_service.get_latest_order(
            customer_id=context.customer_id
        )

    def get_order_status(
        self,
        context: RequestContext,
        order_id: str,
    ) -> OrderStatus | None:

        return self.order_service.get_order_status(
            customer_id=context.customer_id,
            order_id=order_id,
        )

    def get_order_details(
        self,
        context: RequestContext,
        order_id: str,
    ) -> list[OrderItem]:

        return self.order_service.get_order_details(
            customer_id=context.customer_id,
            order_id=order_id,
        )

    def get_order_history(
        self,
        context: RequestContext,
        limit: int = 20,
    ) -> list[OrderHistoryItem]:

        return self.order_service.get_order_history(
            customer_id=context.customer_id,
            limit=limit,
        )

    def search_customer_orders(
        self,
        context: RequestContext,
        search_term: str,
        limit: int = 20,
    ) -> list[CustomerOrderSearchResult]:

        return self.order_service.search_customer_orders(
            customer_id=context.customer_id,
            search_term=search_term,
            limit=limit,
        )

    def get_purchased_products(
        self,
        context: RequestContext,
        limit: int = 50,
    ) -> list[PurchasedProduct]:

        return self.order_service.get_purchased_products(
            customer_id=context.customer_id,
            limit=limit,
        )

    def get_product_purchase_history(
        self,
        context: RequestContext,
        product_query: str,
        limit: int = 20,
    ) -> list[ProductPurchaseHistory]:

        return self.order_service.get_product_purchase_history(
            customer_id=context.customer_id,
            product_query=product_query,
            limit=limit,
        )