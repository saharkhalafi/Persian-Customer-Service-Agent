import time

from app.agents.agent import CustomerSupportAgent
from app.core.cache import TtlLruCache
from app.core.context import RequestContext
from app.retrieval.knowledge_retriever import _SEARCH_CACHE
from app.services.product_metadata import (
    _extract_product_metadata_cached,
    extract_product_metadata,
)
from tests.test_agent_search_products import (
    FakeChat,
    FakeFunctionCall,
    FakeGeminiClient,
    FakeResponse,
    StubTools,
    _agent,
)
from app.tools.product_tools import ProductTools


def test_deterministic_metadata_cache_returns_same_object():
    _extract_product_metadata_cached.cache_clear()
    first = extract_product_metadata("کرم ضد چروک پرایم")
    second = extract_product_metadata("کرم ضد چروک پرایم")
    assert first == second
    assert first is second
    info = _extract_product_metadata_cached.cache_info()
    assert info.hits >= 1


def test_ttl_lru_cache_expires_and_evicts():
    cache = TtlLruCache(maxsize=2, ttl_seconds=0.05)
    cache.set("a", 1)
    assert cache.get("a") == 1
    time.sleep(0.06)
    assert cache.get("a") is None
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.get("c") == 3


def test_knowledge_cache_can_be_cleared():
    _SEARCH_CACHE.clear()
    _SEARCH_CACHE.set(("مرجوعی", 3), [{"chunk_id": 1, "content": "x"}])
    assert _SEARCH_CACHE.get(("مرجوعی", 3))[0]["chunk_id"] == 1
    _SEARCH_CACHE.clear()
    assert _SEARCH_CACHE.get(("مرجوعی", 3)) is None


class RecordingKnowledge(StubTools):
    def __init__(self):
        super().__init__()
        self.started = None
        self.ended = None

    def search_knowledge_base(self, context, query, limit=3):
        self.started = time.perf_counter()
        time.sleep(0.05)
        self.ended = time.perf_counter()
        self.calls.append(("search_knowledge_base", query))
        return []


class RecordingOrders(StubTools):
    def __init__(self):
        super().__init__()
        self.started = None
        self.ended = None

    def get_latest_order(self, context):
        self.started = time.perf_counter()
        time.sleep(0.05)
        self.ended = time.perf_counter()
        self.calls.append(("get_latest_order",))
        return None


def test_independent_tools_run_in_parallel():
    knowledge = RecordingKnowledge()
    orders = RecordingOrders()
    chat = FakeChat(
        [
            FakeResponse(
                function_calls=[
                    FakeFunctionCall("search_knowledge_base", {"query": "مرجوعی"}),
                    FakeFunctionCall("get_latest_order", {}),
                ]
            ),
            FakeResponse(text="پاسخ ترکیبی"),
        ]
    )
    agent = CustomerSupportAgent(
        gemini_client=FakeGeminiClient(chat),
        order_tools=orders,
        customer_tools=StubTools(),
        knowledge_tools=knowledge,
        product_tools=ProductTools(StubTools()),
        conversation_tools=StubTools(),
    )
    started = time.perf_counter()
    result = agent.run(
        user_message="آخرین سفارشم و سیاست مرجوعی",
        context=RequestContext(customer_id="9206288"),
    )
    elapsed = time.perf_counter() - started
    assert [call.name for call in result.tool_calls] == [
        "search_knowledge_base",
        "get_latest_order",
    ]
    assert elapsed < 0.15
    assert knowledge.started is not None and orders.started is not None
    overlap = min(knowledge.ended, orders.ended) - max(knowledge.started, orders.started)
    assert overlap > 0


def test_existing_single_tool_path_still_works():
    from tests.test_agent_search_products import RecordingProductService
    from app.schemas.product_schemas import ProductSearchItem

    service = RecordingProductService(
        [ProductSearchItem(product_name="کرم", brand="Prime")]
    )
    chat = FakeChat(
        [
            FakeResponse(
                function_calls=[
                    FakeFunctionCall("search_products", {"query": "کرم پرایم"})
                ]
            ),
            FakeResponse(text="نتایج"),
        ]
    )
    result = _agent(chat, ProductTools(service)).run(
        user_message="کرم پرایم",
        context=RequestContext(customer_id="9206288"),
    )
    assert [call.name for call in result.tool_calls] == ["search_products"]
    assert service.calls == [{"query": "کرم پرایم", "limit": 10}]
