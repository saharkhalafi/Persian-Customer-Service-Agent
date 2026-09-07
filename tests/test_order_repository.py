from app.core.database import SessionLocal
from app.repositories.order_repository import OrderRepository
from app.services.order_service import OrderService


def main():
    db = SessionLocal()

    try:
        repository = OrderRepository(db)
        service = OrderService(repository)

        result = service.get_customer_order_summary(
            "9206288"
        )

        print(result)

    finally:
        db.close()


if __name__ == "__main__":
    main()