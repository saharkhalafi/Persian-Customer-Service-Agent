from types import SimpleNamespace

from app.agents.agent import CustomerSupportAgent
from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tool_registry import get_tool_declarations
from app.core.config import CONVERSATION_HISTORY_LIMIT
from app.core.context import RequestContext
from app.schemas.product_schemas import ProductSearchItem
from app.tools.product_tools import (
    EMPTY_PRODUCT_SEARCH_MESSAGE,
    ProductTools,
    serialize_product_search_results,
)


PRODUCT_INTENT_QUERIES = [
    "ماساژور شیائومی میخوام",
    "کرم ضد چروک پرایم دارید؟",
    "عطر زنانه زیر ۵۰۰ هزار تومان میخوام",
    "محصولات سالوته رو میخوام",
]

NON_PRODUCT_CASES = [
    ("سفارشم کجاست؟", "get_latest_order"),
    ("چطور رمز عبورم رو بازیابی کنم؟", "search_knowledge_base"),
    ("آیا امکان مرجوعی هست؟", "search_knowledge_base"),
]


class FakeFunctionCall:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class FakeResponse:
    def __init__(self, function_calls=None, text=""):
        self.function_calls = function_calls or []
        self.text = text


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def send_message(self, message):
        self.sent.append(message)
        return self.responses.pop(0)


class FakeGeminiClient:
    def __init__(self, chat: FakeChat):
        self.model = "fake-model"
        self.client = SimpleNamespace(
            chats=SimpleNamespace(create=lambda **_kwargs: chat)
        )


class RecordingProductService:
    def __init__(self, items=None):
        self.items = items or []
        self.calls = []

    def search_products(self, query, limit=10):
        self.calls.append({"query": query, "limit": limit})
        return list(self.items)


class StubTools:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return None

        return method


def _agent(chat: FakeChat, product_tools: ProductTools) -> CustomerSupportAgent:
    conversation = StubTools()
    return CustomerSupportAgent(
        gemini_client=FakeGeminiClient(chat),
        order_tools=StubTools(),
        customer_tools=StubTools(),
        knowledge_tools=StubTools(),
        product_tools=product_tools,
        conversation_tools=conversation,
    )


def _search_declaration():
    return next(
        item
        for item in get_tool_declarations()
        if item.name == "search_products"
    )


def test_search_products_declaration_accepts_only_natural_language_query():
    declaration = _search_declaration()
    properties = declaration.parameters.properties

    assert declaration.parameters.required == ["query"]
    assert list(properties) == ["query"]
    assert "customer_id" not in properties
    assert "filters" not in properties
    assert "sql" not in properties
    assert "limit" not in properties
    description = declaration.description.lower()
    assert "catalog" in description
    assert "sql" in description
    assert "customer_id" in description
    assert "faq" in description


def test_prompt_selects_search_products_for_product_intent_examples():
    for query in PRODUCT_INTENT_QUERIES:
        assert query in SYSTEM_PROMPT
    assert "search_products" in SYSTEM_PROMPT


def test_prompt_does_not_select_search_products_for_order_faq_or_greeting():
    assert "سلام" in SYSTEM_PROMPT
    assert "بدون ابزار" in SYSTEM_PROMPT
    for query, expected_tool in NON_PRODUCT_CASES:
        assert query in SYSTEM_PROMPT
        assert expected_tool in SYSTEM_PROMPT or "ابزار سفارش" in SYSTEM_PROMPT


def test_agent_selects_search_products_for_product_intent_queries():
    service = RecordingProductService(
        [
            ProductSearchItem(
                product_name="ماساژور شیائومی",
                brand="Xiaomi",
                special_price="2500000",
            )
        ]
    )
    tools = ProductTools(service)

    for query in PRODUCT_INTENT_QUERIES:
        service.calls.clear()
        chat = FakeChat(
            [
                FakeResponse(
                    function_calls=[
                        FakeFunctionCall(
                            "search_products",
                            {"query": query},
                        )
                    ]
                ),
                FakeResponse(text="نتایج جستجو"),
            ]
        )
        result = _agent(chat, tools).run(
            user_message=query,
            context=RequestContext(customer_id="9206288"),
        )

        assert [call.name for call in result.tool_calls] == ["search_products"]
        assert service.calls == [{"query": query, "limit": 10}]


def test_agent_does_not_select_search_products_for_order_faq_or_general():
    service = RecordingProductService()
    tools = ProductTools(service)
    scripted = [
        (
            "سفارشم کجاست؟",
            [FakeFunctionCall("get_latest_order", {})],
            "وضعیت سفارش",
        ),
        (
            "چطور رمز عبورم رو بازیابی کنم؟",
            [FakeFunctionCall("search_knowledge_base", {"query": "بازیابی رمز عبور"})],
            "راهنمای بازیابی",
        ),
        (
            "آیا امکان مرجوعی هست؟",
            [FakeFunctionCall("search_knowledge_base", {"query": "مرجوعی"})],
            "سیاست مرجوعی",
        ),
        ("سلام", [], "سلام! چطور می‌توانم کمک کنم؟"),
        ("آب و هوای فردا چطوره؟", [], "امکان بررسی آن اطلاعات را ندارم."),
    ]

    for query, function_calls, answer in scripted:
        service.calls.clear()
        responses = [FakeResponse(function_calls=function_calls, text=answer)]
        if function_calls:
            responses.append(FakeResponse(text=answer))
        result = _agent(FakeChat(responses), tools).run(
            user_message=query,
            context=RequestContext(customer_id="9206288"),
        )

        assert "search_products" not in [call.name for call in result.tool_calls]
        assert service.calls == []


def test_agent_passes_only_natural_language_query_and_ignores_unsafe_args():
    service = RecordingProductService(
        [ProductSearchItem(product_name="کرم پرایم", brand="Prime")]
    )
    tools = ProductTools(service)
    query = "کرم ضد چروک پرایم دارید؟"
    chat = FakeChat(
        [
            FakeResponse(
                function_calls=[
                    FakeFunctionCall(
                        "search_products",
                        {
                            "query": query,
                            "customer_id": "9999",
                            "sql": "SELECT * FROM products",
                            "filters": {"brand": "hacked"},
                            "limit": 3,
                        },
                    )
                ]
            ),
            FakeResponse(text="کرم پرایم موجود است."),
        ]
    )

    result = _agent(chat, tools).run(
        user_message=query,
        context=RequestContext(customer_id="9206288"),
    )

    assert result.tool_calls[0].arguments["query"] == query
    assert "customer_id" not in result.tool_calls[0].arguments
    assert "sql" not in result.tool_calls[0].arguments
    assert service.calls == [{"query": query, "limit": 10}]
    assert "customer_id" not in service.calls[0]
    assert "sql" not in service.calls[0]
    assert "filters" not in service.calls[0]


def test_product_search_results_are_grounded_in_tool_payload():
    items = [
        ProductSearchItem(
            product_code="P1",
            product_name="عطر زنانه",
            brand="Rodier",
            special_price="450000",
            last_status="Enable",
            available_qty="4",
        )
    ]
    payload = serialize_product_search_results(items)

    assert payload["found"] is True
    assert payload["results"] == [
        {
            "product_code": "P1",
            "product_name": "عطر زنانه",
            "brand": "Rodier",
            "special_price": "450000",
            "last_status": "Enable",
            "available_qty": "4",
        }
    ]
    assert "url" not in payload["results"][0]
    assert "discount" not in payload["results"][0]


def test_empty_results_do_not_hallucinate_products():
    service = RecordingProductService([])
    tools = ProductTools(service)
    chat = FakeChat(
        [
            FakeResponse(
                function_calls=[
                    FakeFunctionCall(
                        "search_products",
                        {"query": "عطر ناموجود فضایی"},
                    )
                ]
            ),
            FakeResponse(
                text=(
                    "پیشنهاد می‌کنم عطر شانل با ۳۰٪ تخفیف "
                    "از https://example.com بخرید."
                )
            ),
        ]
    )

    result = _agent(chat, tools).run(
        user_message="عطر ناموجود فضایی",
        context=RequestContext(customer_id="9206288"),
    )

    assert result.tool_calls[0].name == "search_products"
    assert EMPTY_PRODUCT_SEARCH_MESSAGE in result.answer
    assert "شانل" not in result.answer
    assert "۳۰٪" not in result.answer
    assert "https://example.com" not in result.answer


def test_missing_price_url_and_availability_are_not_invented():
    item = ProductSearchItem(
        product_name="کرم ضد چروک پرایم",
        brand="Prime",
    )
    payload = serialize_product_search_results([item])
    result = payload["results"][0]

    assert result == {
        "product_name": "کرم ضد چروک پرایم",
        "brand": "Prime",
    }
    assert "special_price" not in result
    assert "available_qty" not in result
    assert "last_status" not in result
    assert "url" not in result
    assert "product_url" not in result


def test_product_tools_do_not_pass_customer_id_to_search():
    service = RecordingProductService(
        [ProductSearchItem(product_name="سالوته")]
    )
    tools = ProductTools(service)

    tools.search_products(
        context=RequestContext(customer_id="should-not-be-used"),
        query="محصولات سالوته رو میخوام",
    )

    assert service.calls == [
        {"query": "محصولات سالوته رو میخوام", "limit": 10}
    ]


def test_agent_seeds_existing_conversation_history():
    from app.schemas.conversation_schemas import ConversationMessage

    class HistoryTools(StubTools):
        def get_conversation_history(self, context, limit=10):
            self.calls.append(("get_conversation_history", context.customer_id, limit))
            return [
                ConversationMessage(
                    role="user",
                    content="کرم ضدچروک پرایم دارید؟",
                ),
                ConversationMessage(
                    role="assistant",
                    content="بله، چند محصول پرایم پیدا شد.",
                ),
            ]

        def save_turn(self, context, user_message, assistant_message):
            self.calls.append(("save_turn", user_message, assistant_message))

    created = {}

    class CapturingGemini:
        def __init__(self, chat):
            self.model = "fake-model"

            def create(**kwargs):
                created.update(kwargs)
                return chat

            self.client = SimpleNamespace(chats=SimpleNamespace(create=create))

    chat = FakeChat([FakeResponse(text="حدود ۴۵۰ هزار تومان است.")])
    conversation = HistoryTools()
    agent = CustomerSupportAgent(
        gemini_client=CapturingGemini(chat),
        order_tools=StubTools(),
        customer_tools=StubTools(),
        knowledge_tools=StubTools(),
        product_tools=ProductTools(RecordingProductService()),
        conversation_tools=conversation,
    )
    result = agent.run(
        user_message="قیمتش چنده؟",
        context=RequestContext(customer_id="9206288"),
    )

    assert result.answer == "حدود ۴۵۰ هزار تومان است."
    history = created.get("history") or []
    assert len(history) == 2
    assert history[0].role == "user"
    assert "پرایم" in history[0].parts[0].text
    assert history[1].role == "model"
    assert (
        "get_conversation_history",
        "9206288",
        CONVERSATION_HISTORY_LIMIT,
    ) in conversation.calls
