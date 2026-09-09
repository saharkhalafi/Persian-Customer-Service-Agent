from sqlalchemy import text
from sqlalchemy.orm import Session


class OrderRepository:

    def __init__(self, db: Session):
        self.db = db

    # ---------------------------------------------------------
    # ORDER SUMMARY
    # ---------------------------------------------------------

    def get_order_summary(
        self,
        customer_id: str,
    ) -> dict:

        query = text("""
            SELECT
                COUNT(DISTINCT order_id) AS total_orders,

                COUNT(DISTINCT order_id) FILTER (
                    WHERE LOWER(COALESCE(status, '')) IN
                    ('complete', 'completed', 'delivered')
                ) AS successful_orders,

                COUNT(DISTINCT order_id) FILTER (
                    WHERE LOWER(COALESCE(status, '')) IN
                    ('cancelled', 'canceled')
                ) AS cancelled_orders,

                COUNT(*) AS total_items,

                COALESCE(
                    SUM(
                        CASE
                            WHEN final_price ~ '^[0-9]+(\\.[0-9]+)?$'
                            THEN final_price::numeric
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_spent

            FROM public.orders

            WHERE customer_id = :customer_id
        """)

        result = self.db.execute(
            query,
            {"customer_id": customer_id},
        ).mappings().one()

        return dict(result)

    # ---------------------------------------------------------
    # LATEST ORDER
    # ---------------------------------------------------------

    def get_latest_order(
        self,
        customer_id: str,
    ) -> dict | None:

        query = text("""
            SELECT
                order_id,
                MIN(created_at) AS created_at,
                MAX(status) AS status,
                MAX(delivered_date) AS delivered_date,
                MAX(method) AS method,
                MAX(coupon_code) AS coupon_code,
                MAX(delivery_time) AS delivery_time,
                MAX(city) AS city,
                MAX(seller_type) AS seller_type,
                COUNT(*) AS item_count,

                COALESCE(
                    SUM(
                        CASE
                            WHEN final_price ~ '^[0-9]+(\\.[0-9]+)?$'
                            THEN final_price::numeric
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_amount

            FROM public.orders

            WHERE customer_id = :customer_id

            GROUP BY order_id

            ORDER BY
                MIN(created_at) DESC NULLS LAST

            LIMIT 1
        """)

        result = self.db.execute(
            query,
            {"customer_id": customer_id},
        ).mappings().first()

        return dict(result) if result else None

    # ---------------------------------------------------------
    # ORDER STATUS
    # ---------------------------------------------------------

    def get_order_status(
        self,
        customer_id: str,
        order_id: str,
    ) -> dict | None:

        query = text("""
            SELECT
                order_id,
                MAX(status) AS status,
                MIN(created_at) AS created_at,
                MAX(delivered_date) AS delivered_date,
                MAX(delivery_time) AS delivery_time,
                MAX(method) AS method,
                MAX(city) AS city

            FROM public.orders

            WHERE
                customer_id = :customer_id
                AND order_id = :order_id

            GROUP BY order_id
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "order_id": order_id,
            },
        ).mappings().first()

        return dict(result) if result else None

    # ---------------------------------------------------------
    # ORDER DETAILS
    # ---------------------------------------------------------

    def get_order_details(
        self,
        customer_id: str,
        order_id: str,
    ) -> list[dict]:

        query = text("""
            SELECT
                order_id,
                created_at,
                status,
                delivered_date,
                method,
                coupon_code,
                delivery_time,
                city,
                seller_id,
                seller_type,
                product_id,
                order_items_name AS product_name,
                order_items_sku AS sku,
                brand_name,
                category_level1,
                category_level2,
                invoiced_qty,
                ordered_qty,
                refund_qty,
                shipped_qty,
                sale_price,
                final_price,
                color,
                material

            FROM public.orders

            WHERE
                customer_id = :customer_id
                AND order_id = :order_id

            ORDER BY created_at ASC NULLS LAST
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "order_id": order_id,
            },
        ).mappings()

        return [dict(row) for row in result]

    # ---------------------------------------------------------
    # ORDER HISTORY
    # ---------------------------------------------------------

    def get_order_history(
        self,
        customer_id: str,
        limit: int = 20,
    ) -> list[dict]:

        limit = min(max(limit, 1), 50)

        query = text("""
            SELECT
                order_id,
                MIN(created_at) AS created_at,
                MAX(status) AS status,
                MAX(delivered_date) AS delivered_date,
                MAX(delivery_time) AS delivery_time,
                MAX(method) AS method,
                MAX(city) AS city,
                COUNT(*) AS item_count,

                COALESCE(
                    SUM(
                        CASE
                            WHEN final_price ~ '^[0-9]+(\\.[0-9]+)?$'
                            THEN final_price::numeric
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_amount

            FROM public.orders

            WHERE customer_id = :customer_id

            GROUP BY order_id

            ORDER BY
                MIN(created_at) DESC NULLS LAST

            LIMIT :limit
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in result]

    # ---------------------------------------------------------
    # SEARCH CUSTOMER ORDERS
    # ---------------------------------------------------------

    def search_customer_orders(
        self,
        customer_id: str,
        search_term: str,
        limit: int = 20,
    ) -> list[dict]:

        limit = min(max(limit, 1), 50)

        query = text("""
            SELECT
                order_id,
                MIN(created_at) AS created_at,
                MAX(status) AS status,
                MAX(delivered_date) AS delivered_date,
                MAX(delivery_time) AS delivery_time,
                MAX(city) AS city,
                COUNT(*) AS item_count,

                STRING_AGG(
                    DISTINCT order_items_name,
                    ', '
                ) AS products

            FROM public.orders

            WHERE
                customer_id = :customer_id
                AND (
                    order_id ILIKE :term
                    OR order_items_name ILIKE :term
                    OR order_items_sku ILIKE :term
                    OR product_id ILIKE :term
                    OR brand_name ILIKE :term
                    OR category_level1 ILIKE :term
                    OR category_level2 ILIKE :term
                    OR status ILIKE :term
                )

            GROUP BY order_id

            ORDER BY
                MIN(created_at) DESC NULLS LAST

            LIMIT :limit
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "term": f"%{search_term}%",
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in result]

    # ---------------------------------------------------------
    # PURCHASED PRODUCTS
    # ---------------------------------------------------------

    def get_purchased_products(
        self,
        customer_id: str,
        limit: int = 50,
    ) -> list[dict]:

        limit = min(max(limit, 1), 100)

        query = text("""
            SELECT
                order_id,
                created_at,
                order_items_name AS product_name,
                order_items_sku AS sku,
                product_id,
                brand_name,
                category_level1,
                category_level2,
                ordered_qty,
                final_price

            FROM public.orders

            WHERE customer_id = :customer_id

            ORDER BY
                created_at DESC NULLS LAST

            LIMIT :limit
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in result]

    # ---------------------------------------------------------
    # PRODUCT PURCHASE HISTORY
    # ---------------------------------------------------------

    def get_product_purchase_history(
        self,
        customer_id: str,
        product_query: str,
        limit: int = 20,
    ) -> list[dict]:

        limit = min(max(limit, 1), 50)

        query = text("""
            SELECT
                order_items_name AS product_name,
                order_items_sku AS sku,
                product_id,
                brand_name,

                COUNT(DISTINCT order_id) AS order_count,

                COALESCE(
                    SUM(
                        CASE
                            WHEN ordered_qty ~ '^[0-9]+(\\.[0-9]+)?$'
                            THEN ordered_qty::numeric
                            ELSE 0
                        END
                    ),
                    0
                ) AS total_quantity,

                MAX(created_at) AS last_purchase_date

            FROM public.orders

            WHERE
                customer_id = :customer_id
                AND (
                    order_items_name ILIKE :term
                    OR order_items_sku ILIKE :term
                    OR product_id ILIKE :term
                    OR brand_name ILIKE :term
                )

            GROUP BY
                order_items_name,
                order_items_sku,
                product_id,
                brand_name

            ORDER BY
                MAX(created_at) DESC NULLS LAST

            LIMIT :limit
        """)

        result = self.db.execute(
            query,
            {
                "customer_id": customer_id,
                "term": f"%{product_query}%",
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in result]