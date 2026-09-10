from sqlalchemy import text
from sqlalchemy.orm import Session


class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict]:
        limit = min(max(limit, 1), 20)

        sql = text("""
            SELECT
                product_code,
                product_name,
                brand,
                type,
                sku,
                category_level1,
                category_level2,
                last_status,
                available_qty,
                special_price,
                color,
                size
            FROM public.products
            WHERE
                product_name ILIKE :term
                OR brand ILIKE :term
                OR type ILIKE :term
                OR sku ILIKE :term
                OR product_code ILIKE :term
                OR category_level1 ILIKE :term
                OR category_level2 ILIKE :term
                OR color ILIKE :term
            ORDER BY
                CASE
                    WHEN last_status = 'Enable' THEN 0
                    ELSE 1
                END,
                product_name ASC NULLS LAST
            LIMIT :limit
        """)

        result = self.db.execute(
            sql,
            {
                "term": f"%{query}%",
                "limit": limit,
            },
        ).mappings()

        return [dict(row) for row in result]
