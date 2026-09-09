from sqlalchemy import text
from sqlalchemy.orm import Session


class CustomerRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_customer_profile(
        self,
        customer_id: str,
        fields: list[str] | None = None,
    ) -> dict | None:

        allowed_fields = {
            "customer_code",
            "customer_group",
            "customer_type",
            "city",
            "customer_name",
            "gender",
            "mobile",
            "email",
            "created_date_persian",
            "first_purchase",
            "last_purchase",
            "success_ordered_cnt",
            "success_order_price",
            "item_cnt",
        }

        if fields:
            selected_fields = [
                field
                for field in fields
                if field in allowed_fields
            ]
        else:
            selected_fields = list(allowed_fields)

        if not selected_fields:
            selected_fields = list(allowed_fields)

        columns = ", ".join(selected_fields)

        query = text(f"""
            SELECT
                {columns}
            FROM public.users
            WHERE customer_code = :customer_id
            LIMIT 1
        """)

        result = self.db.execute(
            query,
            {"customer_id": str(customer_id)},
        ).mappings().first()

        if not result:
            return None

        return dict(result)