from pydantic import BaseModel


class CustomerProfile(BaseModel):
    customer_code: str | None = None
    customer_group: str | None = None
    customer_type: str | None = None
    city: str | None = None
    customer_name: str | None = None
    gender: str | None = None
    mobile: str | None = None
    email: str | None = None
    created_date_persian: str | None = None
    first_purchase: str | None = None
    last_purchase: str | None = None
    success_ordered_cnt: int | None = None
    success_order_price: float | None = None
    item_cnt: int | None = None