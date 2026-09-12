from app.core.observability import traced
from app.repositories.order_repository import OrderRepository
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


class OrderService:

    def __init__(
        self,
        order_repository: OrderRepository,
    ):
        self.order_repository = order_repository

    @traced("service_operation", layer="order", operation="get_customer_order_summary")
    def get_customer_order_summary(
        self,
        customer_id: str,
    ) -> CustomerOrderSummary:

        summary = self.order_repository.get_order_summary(
            customer_id=customer_id
        )

        return CustomerOrderSummary(
            customer_id=customer_id,
            total_orders=summary["total_orders"] or 0,
            successful_orders=summary["successful_orders"] or 0,
            cancelled_orders=summary["cancelled_orders"] or 0,
            total_items=summary["total_items"] or 0,
            total_spent=float(
                summary["total_spent"] or 0
            ),
        )

    @traced("service_operation", layer="order", operation="get_latest_order")
    def get_latest_order(
        self,
        customer_id: str,
    ) -> LatestOrder | None:

        result = self.order_repository.get_latest_order(
            customer_id=customer_id
        )

        if not result:
            return None

        return LatestOrder(
            order_id=result.get("order_id"),
            created_at=result.get("created_at"),
            status=result.get("status"),
            delivered_date=result.get("delivered_date"),
            method=result.get("method"),
            coupon_code=result.get("coupon_code"),
            delivery_time=result.get("delivery_time"),
            city=result.get("city"),
            seller_type=result.get("seller_type"),
            item_count=result.get("item_count") or 0,
            total_amount=float(
                result.get("total_amount") or 0
            ),
        )

    @traced("service_operation", layer="order", operation="get_order_status")
    def get_order_status(
        self,
        customer_id: str,
        order_id: str,
    ) -> OrderStatus | None:

        result = self.order_repository.get_order_status(
            customer_id=customer_id,
            order_id=order_id,
        )

        if not result:
            return None

        return OrderStatus(
            order_id=result["order_id"],
            status=result.get("status"),
            created_at=result.get("created_at"),
            delivered_date=result.get("delivered_date"),
            delivery_time=result.get("delivery_time"),
            method=result.get("method"),
            city=result.get("city"),
        )

    @traced("service_operation", layer="order", operation="get_order_details")
    def get_order_details(
        self,
        customer_id: str,
        order_id: str,
    ) -> list[OrderItem]:

        rows = self.order_repository.get_order_details(
            customer_id=customer_id,
            order_id=order_id,
        )

        return [
            OrderItem(
                order_id=row.get("order_id"),
                created_at=row.get("created_at"),
                status=row.get("status"),
                product_id=row.get("product_id"),
                product_name=row.get("product_name"),
                sku=row.get("sku"),
                brand_name=row.get("brand_name"),
                category_level1=row.get("category_level1"),
                category_level2=row.get("category_level2"),
                ordered_qty=row.get("ordered_qty"),
                invoiced_qty=row.get("invoiced_qty"),
                refund_qty=row.get("refund_qty"),
                shipped_qty=row.get("shipped_qty"),
                sale_price=row.get("sale_price"),
                final_price=row.get("final_price"),
                color=row.get("color"),
                material=row.get("material"),
            )
            for row in rows
        ]

    @traced("service_operation", layer="order", operation="get_order_history")
    def get_order_history(
        self,
        customer_id: str,
        limit: int = 20,
    ) -> list[OrderHistoryItem]:

        rows = self.order_repository.get_order_history(
            customer_id=customer_id,
            limit=limit,
        )

        return [
            OrderHistoryItem(
                order_id=row["order_id"],
                created_at=row.get("created_at"),
                status=row.get("status"),
                delivered_date=row.get("delivered_date"),
                delivery_time=row.get("delivery_time"),
                method=row.get("method"),
                city=row.get("city"),
                item_count=row.get("item_count") or 0,
                total_amount=float(
                    row.get("total_amount") or 0
                ),
            )
            for row in rows
        ]

    @traced("service_operation", layer="order", operation="search_customer_orders")
    def search_customer_orders(
        self,
        customer_id: str,
        search_term: str,
        limit: int = 20,
    ) -> list[CustomerOrderSearchResult]:

        rows = self.order_repository.search_customer_orders(
            customer_id=customer_id,
            search_term=search_term,
            limit=limit,
        )

        return [
            CustomerOrderSearchResult(
                order_id=row["order_id"],
                created_at=row.get("created_at"),
                status=row.get("status"),
                delivered_date=row.get("delivered_date"),
                delivery_time=row.get("delivery_time"),
                city=row.get("city"),
                item_count=row.get("item_count") or 0,
                products=row.get("products"),
            )
            for row in rows
        ]

    @traced("service_operation", layer="order", operation="get_purchased_products")
    def get_purchased_products(
        self,
        customer_id: str,
        limit: int = 50,
    ) -> list[PurchasedProduct]:

        rows = self.order_repository.get_purchased_products(
            customer_id=customer_id,
            limit=limit,
        )

        return [
            PurchasedProduct(
                order_id=row.get("order_id"),
                created_at=row.get("created_at"),
                product_name=row.get("product_name"),
                sku=row.get("sku"),
                product_id=row.get("product_id"),
                brand_name=row.get("brand_name"),
                category_level1=row.get("category_level1"),
                category_level2=row.get("category_level2"),
                ordered_qty=row.get("ordered_qty"),
                final_price=row.get("final_price"),
            )
            for row in rows
        ]

    @traced("service_operation", layer="order", operation="get_product_purchase_history")
    def get_product_purchase_history(
        self,
        customer_id: str,
        product_query: str,
        limit: int = 20,
    ) -> list[ProductPurchaseHistory]:

        rows = self.order_repository.get_product_purchase_history(
            customer_id=customer_id,
            product_query=product_query,
            limit=limit,
        )

        return [
            ProductPurchaseHistory(
                product_name=row.get("product_name"),
                sku=row.get("sku"),
                product_id=row.get("product_id"),
                brand_name=row.get("brand_name"),
                order_count=row.get("order_count") or 0,
                total_quantity=float(
                    row.get("total_quantity") or 0
                ),
                last_purchase_date=row.get(
                    "last_purchase_date"
                ),
            )
            for row in rows
        ]