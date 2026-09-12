from sqlalchemy.orm import Session

from app.agents.agent import CustomerSupportAgent
from app.core.gemini import GeminiClient

from app.repositories.order_repository import OrderRepository
from app.repositories.customer_repository import CustomerRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.product_repository import ProductRepository
from app.repositories.conversation_repository import ConversationRepository

from app.services.order_service import OrderService
from app.services.customer_service import CustomerService
from app.services.knowledge_service import KnowledgeService
from app.services.product_llm_config import load_product_llm_config
from app.services.product_llm_ranker import ProductLlmRanker, gemini_generate_json
from app.services.product_service import ProductService
from app.services.conversation_service import ConversationService

from app.tools.order_tools import OrderTools
from app.tools.customer_tools import CustomerTools
from app.tools.knowledge_tools import KnowledgeTools
from app.tools.product_tools import ProductTools
from app.tools.conversation_tools import ConversationTools


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

    # Knowledge base
    knowledge_repository = KnowledgeRepository(gemini)
    knowledge_service = KnowledgeService(knowledge_repository)
    knowledge_tools = KnowledgeTools(knowledge_service)

    # Product catalog
    product_repository = ProductRepository(db)
    product_llm_config = load_product_llm_config()
    product_service = ProductService(
        product_repository,
        llm_ranker=ProductLlmRanker(
            generate_json=gemini_generate_json(gemini),
            config=product_llm_config,
        ),
        llm_config=product_llm_config,
    )
    product_tools = ProductTools(product_service)

    # Conversation memory
    conversation_repository = ConversationRepository(db)
    conversation_service = ConversationService(
        conversation_repository
    )
    conversation_tools = ConversationTools(
        conversation_service
    )

    return CustomerSupportAgent(
        gemini_client=gemini,
        order_tools=order_tools,
        customer_tools=customer_tools,
        knowledge_tools=knowledge_tools,
        product_tools=product_tools,
        conversation_tools=conversation_tools,
    )