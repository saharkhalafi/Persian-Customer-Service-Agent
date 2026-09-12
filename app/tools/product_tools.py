from app.core.context import RequestContext
from app.schemas.product_schemas import ProductSearchItem
from app.services.product_service import ProductService

EMPTY_PRODUCT_SEARCH_MESSAGE = "محصول مناسبی پیدا نکردم"


def serialize_product_search_results(
    items: list[ProductSearchItem],
) -> dict:
    results = [
        item.model_dump(exclude_none=True)
        for item in items
    ]
    payload = {
        "found": bool(results),
        "results": results,
    }
    if not results:
        payload["message"] = (
            "No suitable products were found. "
            "Reply briefly in Persian that no suitable product was found. "
            "Do not invent products, prices, discounts, availability, or URLs."
        )
    return payload


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
