from sqlalchemy import text
from sqlalchemy.orm import Session


class OrderRepository:

    def __init__(self, db: Session):
        self.db = db

    def get_order_summary(self, customer_id: str) -> dict:
        query = text("""
            SELECT
                COUNT(DISTINCT order_id) AS total_orders,

                COUNT(DISTINCT order_id) FILTER (
                    WHERE LOWER(status) IN ('complete', 'completed', 'delivered')
                ) AS successful_orders,

                COUNT(DISTINCT order_id) FILTER (
                    WHERE LOWER(status) IN ('cancelled', 'canceled')
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
            {"customer_id": customer_id}
        ).mappings().one()

        return dict(result)