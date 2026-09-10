from app.core.context import RequestContext
from app.schemas.product_schemas import ProductSearchItem
from app.services.product_service import ProductService


class ProductTools:
    def __init__(
        self,
        service: ProductService,
    ):
        self.service = service

    def search_products(
        self,
        context: RequestContext,
        query: str,
        limit: int = 10,
    ) -> list[ProductSearchItem]:
        _ = context
        return self.service.search_products(
            query=query,
            limit=limit,
        )
