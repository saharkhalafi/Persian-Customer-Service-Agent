from pydantic import BaseModel


class CustomerOrderSummary(BaseModel):
    customer_id: str
    total_orders: int
    successful_orders: int
    cancelled_orders: int
    total_items: int
    total_spent: float


class LatestOrder(BaseModel):
    order_id: str | None = None
    created_at: str | None = None
    status: str | None = None
    delivered_date: str | None = None
    method: str | None = None
    coupon_code: str | None = None
    delivery_time: str | None = None
    city: str | None = None
    seller_type: str | None = None
    item_count: int = 0
    total_amount: float = 0


class OrderStatus(BaseModel):
    order_id: str
    status: str | None = None
    created_at: str | None = None
    delivered_date: str | None = None
    delivery_time: str | None = None
    method: str | None = None
    city: str | None = None


class OrderItem(BaseModel):
    order_id: str | None = None
    created_at: str | None = None
    status: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    sku: str | None = None
    brand_name: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    ordered_qty: str | None = None
    invoiced_qty: str | None = None
    refund_qty: str | None = None
    shipped_qty: str | None = None
    sale_price: str | None = None
    final_price: str | None = None
    color: str | None = None
    material: str | None = None


class OrderHistoryItem(BaseModel):
    order_id: str
    created_at: str | None = None
    status: str | None = None
    delivered_date: str | None = None
    delivery_time: str | None = None
    method: str | None = None
    city: str | None = None
    item_count: int = 0
    total_amount: float = 0


class PurchasedProduct(BaseModel):
    order_id: str | None = None
    created_at: str | None = None
    product_name: str | None = None
    sku: str | None = None
    product_id: str | None = None
    brand_name: str | None = None
    category_level1: str | None = None
    category_level2: str | None = None
    ordered_qty: str | None = None
    final_price: str | None = None


class ProductPurchaseHistory(BaseModel):
    product_name: str | None = None
    sku: str | None = None
    product_id: str | None = None
    brand_name: str | None = None
    order_count: int = 0
    total_quantity: float = 0
    last_purchase_date: str | None = None


class CustomerOrderSearchResult(BaseModel):
    order_id: str
    created_at: str | None = None
    status: str | None = None
    delivered_date: str | None = None
    delivery_time: str | None = None
    city: str | None = None
    item_count: int = 0
    products: str | None = None