from pydantic import BaseModel


class ProductSearchItem(BaseModel):
    product_code: str | None = None
    product_name: str | None = None
    brand: str | None = None
    type: str | None = None
    sku: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    last_status: str | None = None
    available_qty: str | None = None
    special_price: str | None = None
    color: str | None = None
    size: str | None = None
