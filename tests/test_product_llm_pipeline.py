from app.services.product_llm_config import ProductLlmConfig
from app.services.product_llm_ranker import (
    LlmJsonResult,
    ProductLlmRanker,
    apply_rerank_order,
    merge_llm_understanding,
    needs_query_understanding,
    needs_rerank,
)
from app.services.product_metadata import (
    ProductSearchMetadata,
    extract_product_metadata,
    search_tokens_for_metadata,
)
from app.services.product_service import ProductService


class FakeRepository:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or [
            {
                "product_code": "111",
                "product_name": "Salute Cream",
                "brand": "Salute",
                "type": "کرم دست",
                "sku": "s1",
                "category_level1": "آرایشی",
                "category_level2": "مراقبت",
                "last_status": "Enable",
                "available_qty": "3",
                "special_price": "100000",
                "sum_of_price": "100000",
                "color": "سفید",
                "size": None,
                "gender": "زنانه",
            }
        ]
        self.last_call: dict | None = None

    def search_products(self, **kwargs):
        self.last_call = kwargs
        return list(self.rows)


class ScriptedLlm:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.calls = 0

    def __call__(self, prompt: str, schema: dict, timeout_seconds: float):
        self.prompts.append(prompt)
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, LlmJsonResult):
            return response
        return LlmJsonResult(data=response, input_tokens=10, output_tokens=5)


def _config(**overrides) -> ProductLlmConfig:
    values = dict(
        query_understanding_enabled=True,
        rerank_enabled=True,
        candidate_k=20,
        final_k=10,
        confidence_threshold=0.7,
        metadata_confidence_threshold=0.75,
        timeout_seconds=5.0,
        metadata_fallback_enabled=False,
    )
    values.update(overrides)
    return ProductLlmConfig(**values)


def test_deterministic_query_does_not_call_llm():
    llm = ScriptedLlm([{"confidence": 0.9, "attributes": [], "price_intent": "none"}])
    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("محصولات سالوته")
    assert llm.calls == 0


def test_signature_and_xiaomi_skip_llm():
    llm = ScriptedLlm([])
    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("پنکک Signature")
    service.search_products("ماساژور شیائومی")
    assert llm.calls == 0


def test_low_confidence_query_calls_query_understanding():
    query = "یه چیزی برای موهای خشک میخوام که خیلی گرون نباشه"
    meta = extract_product_metadata(query)
    tokens = search_tokens_for_metadata(query, meta)
    assert needs_query_understanding(query, meta, tokens)

    llm = ScriptedLlm(
        [
            {
                "category_or_type": "ماسک مو",
                "brand": None,
                "attributes": ["موهای خشک"],
                "price_intent": "budget",
                "price_min": None,
                "price_max": None,
                "confidence": 0.82,
            }
        ]
    )
    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(rerank_enabled=False),
    )
    outcome = service.search_with_debug(query)
    assert outcome.debug.llm_query_understanding_used is True
    assert outcome.debug.llm_rerank_used is False
    assert llm.calls == 1


def test_llm_cannot_override_hard_brand_or_price():
    deterministic = ProductSearchMetadata(brand="Note", price_max=300000)
    merged = merge_llm_understanding(
        deterministic,
        {
            "brand": "Xiaomi",
            "price_max": 1,
            "category_or_type": "ماساژور",
            "attributes": [],
            "price_intent": "budget",
            "confidence": 0.99,
        },
        "محصولات نوت زیر 300هزارتومان",
        0.7,
    )
    assert merged.metadata.brand == "Note"
    assert merged.metadata.price_max == 300000


def test_llm_does_not_invent_price_from_budget_intent():
    query = "یه چیزی برای موهای خشک میخوام که خیلی گرون نباشه"
    merged = merge_llm_understanding(
        extract_product_metadata(query),
        {
            "category_or_type": "ماسک مو",
            "price_max": 200000,
            "attributes": ["موهای خشک"],
            "price_intent": "budget",
            "confidence": 0.9,
        },
        query,
        0.7,
    )
    assert merged.metadata.price_max is None


def test_low_confidence_leaves_fields_null():
    merged = merge_llm_understanding(
        ProductSearchMetadata(),
        {
            "brand": "Salute",
            "category_or_type": "ماسک مو",
            "attributes": [],
            "price_intent": "none",
            "confidence": 0.2,
        },
        "یه چیزی خوب",
        0.7,
    )
    assert merged.metadata.brand is None
    assert merged.metadata.category is None


def test_repository_receives_only_service_filters_and_candidate_limit():
    llm = ScriptedLlm(
        [
            {
                "results": [
                    {"product_code": "222", "score": 0.9, "reason": "closer"},
                    {"product_code": "111", "score": 0.1, "reason": "less"},
                ]
            }
        ]
    )
    rows = [
        {
            "product_code": "111",
            "product_name": "Xiaomi A",
            "brand": "Xiaomi",
            "type": "ماساژور",
            "sum_of_price": "1500000",
            "color": None,
            "category_level1": None,
            "category_level2": None,
            "sku": None,
            "last_status": "Enable",
            "available_qty": "1",
            "special_price": "1500000",
            "size": None,
            "gender": None,
        },
        {
            "product_code": "222",
            "product_name": "Xiaomi Neck",
            "brand": "Xiaomi",
            "type": "ماساژور",
            "sum_of_price": "1800000",
            "color": None,
            "category_level1": None,
            "category_level2": None,
            "sku": None,
            "last_status": "Enable",
            "available_qty": "1",
            "special_price": "1800000",
            "size": None,
            "gender": None,
        },
    ]
    repo = FakeRepository(rows)
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(query_understanding_enabled=False, candidate_k=20),
    )
    query = "یه ماساژور خوب برای گردن میخوام"
    assert needs_rerank(
        query,
        extract_product_metadata(query),
        search_tokens_for_metadata(query, extract_product_metadata(query)),
        2,
    )
    outcome = service.search_with_debug(query, limit=5)
    assert repo.last_call is not None
    assert repo.last_call["limit"] == 20
    assert repo.last_call["filters"].category == "ماساژور"
    assert "product_code" in llm.prompts[0]
    assert "222" in llm.prompts[0]
    assert outcome.items[0].product_code == "222"
    assert {item.product_code for item in outcome.items} <= {"111", "222"}


def test_llm_cannot_introduce_unknown_product_code():
    ranked = apply_rerank_order(
        [{"product_code": "111", "product_name": "A"}],
        [
            {"product_code": "999", "score": 1, "reason": "invented"},
            {"product_code": "111", "score": 0.2, "reason": "ok"},
        ],
    )
    assert [row["product_code"] for row in ranked] == ["111"]


def test_invalid_llm_json_falls_back_to_classical_order():
    llm = ScriptedLlm([ValueError("invalid json")])
    rows = [
        {"product_code": "111", "product_name": "A", "brand": "Xiaomi", "type": "ماساژور"},
        {"product_code": "222", "product_name": "B", "brand": "Xiaomi", "type": "ماساژور"},
    ]
    service = ProductService(
        FakeRepository(rows),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(query_understanding_enabled=False),
    )
    outcome = service.search_with_debug("یه ماساژور خوب برای گردن میخوام")
    assert [item.product_code for item in outcome.items] == ["111", "222"]
    assert outcome.debug.fallback_reason == "rerank_error"
    assert outcome.debug.llm_rerank_used is False


def test_llm_timeout_falls_back_to_classical_order():
    llm = ScriptedLlm([TimeoutError("timed out")])
    rows = [
        {"product_code": "111", "product_name": "A", "brand": "Xiaomi", "type": "ماساژور"},
        {"product_code": "222", "product_name": "B", "brand": "Xiaomi", "type": "ماساژور"},
    ]
    service = ProductService(
        FakeRepository(rows),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(query_understanding_enabled=False),
    )
    outcome = service.search_with_debug("یه ماساژور خوب برای گردن میخوام")
    assert [item.product_code for item in outcome.items] == ["111", "222"]
    assert outcome.debug.fallback_reason == "rerank_timeout"


def test_rerank_preserves_candidate_membership():
    llm = ScriptedLlm(
        [
            {
                "results": [
                    {"product_code": "222", "score": 0.9, "reason": "neck"},
                    {"product_code": "999", "score": 0.8, "reason": "hallucinated"},
                    {"product_code": "111", "score": 0.1, "reason": "other"},
                ]
            }
        ]
    )
    rows = [
        {"product_code": "111", "product_name": "A", "brand": "Xiaomi", "type": "ماساژور"},
        {"product_code": "222", "product_name": "B", "brand": "Xiaomi", "type": "ماساژور"},
    ]
    service = ProductService(
        FakeRepository(rows),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(query_understanding_enabled=False),
    )
    outcome = service.search_with_debug("یه ماساژور خوب برای گردن میخوام")
    codes = [item.product_code for item in outcome.items]
    assert codes == ["222", "111"]
    assert "999" not in codes


def test_llm_query_understanding_cannot_replace_existing_brand():
    query = "یه چیزی از سالوته برای موهای خشک"
    llm = ScriptedLlm(
        [
            {
                "brand": "Xiaomi",
                "category_or_type": "ماسک مو",
                "attributes": ["موهای خشک"],
                "price_intent": "none",
                "confidence": 0.95,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(rerank_enabled=False),
    )
    outcome = service.search_with_debug(query)
    assert outcome.debug.llm_query_understanding_used is True
    assert repo.last_call["filters"].brand == "Salute"
    assert repo.last_call["filters"].brand != "Xiaomi"


def test_hard_filters_sent_to_repository_not_llm_decision():
    llm = ScriptedLlm(
        [
            {
                "brand": "Xiaomi",
                "price_max": 1,
                "attributes": [],
                "price_intent": "none",
                "confidence": 0.99,
            }
        ]
    )
    repo = FakeRepository()
    query = "محصولات نوت زیر 300هزارتومان"
    # This query is deterministic, so LLM should not be called at all.
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products(query)
    assert llm.calls == 0
    assert repo.last_call["filters"].brand == "Note"
    assert repo.last_call["filters"].price_max == 300000
