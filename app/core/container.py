from sqlalchemy.orm import Session

from app.agents.agent import CustomerSupportAgent
from app.core.gemini import GeminiClient

from app.repositories.order_repository import OrderRepository
from app.repositories.customer_repository import CustomerRepository

from app.services.order_service import OrderService
from app.services.customer_service import CustomerService

from app.tools.order_tools import OrderTools
from app.tools.customer_tools import CustomerTools


def build_agent(
    db: Session,
    gemini: GeminiClient,
) -> CustomerSupportAgent:

    # Orders
    order_repository = OrderRepository(db)
    order_service = OrderService(order_repository)
    order_tools = OrderTools(order_service)

    # Customer
    customer_repository = CustomerRepository(db)
    customer_service = CustomerService(customer_repository)
    customer_tools = CustomerTools(customer_service)

    return CustomerSupportAgent(
        gemini_client=gemini,
        order_tools=order_tools,
        customer_tools=customer_tools,
    )