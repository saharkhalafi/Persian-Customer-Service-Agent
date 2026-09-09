from app.core.database import SessionLocal
from app.core.gemini import GeminiClient
from app.repositories.order_repository import OrderRepository
from app.services.order_service import OrderService
from app.tools.order_tools import OrderTools
from app.agents.agent import CustomerSupportAgent
from app.core.context import RequestContext

def main():
    db = SessionLocal()

    try:
        repository = OrderRepository(db)
        service = OrderService(repository)
        tools = OrderTools(service)

        gemini = GeminiClient()

        agent = CustomerSupportAgent(
            gemini_client=gemini,
            order_tools=tools,
        )

        context = RequestContext(
            customer_id="9206288")

        answer = agent.run(
            user_message="جمعاً چقدر خرید کردم؟",
            context=context,)

        print("\nAgent:")
        print(answer)

    finally:
        db.close()


if __name__ == "__main__":
    main()