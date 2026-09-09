from app.core.database import SessionLocal
from app.repositories.order_repository import OrderRepository
from app.services.order_service import OrderService
from app.tools.order_tools import OrderTools


def main():
    db = SessionLocal()

    try:
        repository = OrderRepository(db)
        service = OrderService(repository)
        tools = OrderTools(service)

        result = tools.get_customer_order_summary(
            customer_id="9340018"
        )

        print("\nTool result:")
        print(result)

    finally:
        db.close()


if __name__ == "__main__":
    main()