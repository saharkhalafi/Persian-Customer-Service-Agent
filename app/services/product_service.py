from app.repositories.product_repository import ProductRepository
from app.schemas.product_schemas import ProductSearchItem


class ProductService:
    def __init__(
        self,
        repository: ProductRepository,
    ):
        self.repository = repository

    def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ProductSearchItem]:
        query = query.strip()[:200]

        if not query:
            return []

        limit = min(max(limit, 1), 20)

        rows = self.repository.search_products(
            query=query,
            limit=limit,
        )

        return [
            ProductSearchItem(
                product_code=row.get("product_code"),
                product_name=row.get("product_name"),
                brand=row.get("brand"),
                type=row.get("type"),
                sku=row.get("sku"),
                category_level1=row.get("category_level1"),
                category_level2=row.get("category_level2"),
                last_status=row.get("last_status"),
                available_qty=row.get("available_qty"),
                special_price=row.get("special_price"),
                color=row.get("color"),
                size=row.get("size"),
            )
            for row in rows
        ]
