from app.repositories.customer_repository import CustomerRepository
from app.schemas.customer_schemas import CustomerProfile


class CustomerService:
    def __init__(
        self,
        repository: CustomerRepository,
    ):
        self.repository = repository

    def get_customer_profile(
        self,
        customer_id: str,
        fields: list[str] | None = None,
    ) -> CustomerProfile | None:

        data = self.repository.get_customer_profile(
            customer_id=customer_id,
            fields=fields,
        )

        if not data:
            return None

        return CustomerProfile(**data)