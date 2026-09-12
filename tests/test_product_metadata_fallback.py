from app.services.product_llm_config import ProductLlmConfig
from app.services.product_llm_ranker import LlmJsonResult, ProductLlmRanker, merge_llm_understanding
from app.services.product_metadata import extract_product_metadata, search_tokens_for_metadata
from app.services.product_metadata_confidence import (
    AMBIGUOUS_CATEGORY,
    COLLOQUIAL_NO_METADATA,
    IMPLICIT_USE_NO_CATEGORY,
    PARTIAL_PRODUCT_PHRASE,
    UNKNOWN_BRAND_ALIAS,
    assess_metadata_confidence,
    possible_unknown_brand_alias,
)
from app.services.product_service import ProductService


class FakeRepository:
    def __init__(self, rows: list[dict] | None = None):
        self.rows = rows or [
            {
                "product_code": "111",
                "product_name": "Defacto Hand Cream",
                "brand": "Defacto",
                "type": "کرم دست",
                "sku": "s1",
                "category_level1": "مراقبت",
                "category_level2": None,
                "last_status": "Enable",
                "available_qty": "3",
                "special_price": "100000",
                "sum_of_price": "100000",
                "color": None,
                "size": None,
                "gender": None,
            }
        ]
        self.last_call: dict | None = None

    def search_products(self, **kwargs):
        self.last_call = kwargs
        return list(self.rows)


class ScriptedLlm:
    def __init__(self, responses: list):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt: str, schema: dict, timeout_seconds: float):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, LlmJsonResult):
            return response
        return LlmJsonResult(data=response, input_tokens=8, output_tokens=4)


def _config(**overrides) -> ProductLlmConfig:
    values = dict(
        query_understanding_enabled=False,
        rerank_enabled=False,
        metadata_fallback_enabled=True,
        candidate_k=20,
        final_k=10,
        confidence_threshold=0.7,
        metadata_confidence_threshold=0.75,
        timeout_seconds=5.0,
    )
    values.update(overrides)
    return ProductLlmConfig(**values)


def _assess(query: str):
    metadata = extract_product_metadata(query)
    tokens = search_tokens_for_metadata(query, metadata)
    return assess_metadata_confidence(query, metadata, tokens), metadata, tokens


def test_obvious_queries_are_high_confidence():
    for query in ("رنگ مو", "کرم دست", "لاک ناخن مریدا", "محصولات سالوته"):
        assessment, _, _ = _assess(query)
        assert assessment.needs_llm is False, query
        assert assessment.score >= 0.75, query


def test_leftover_fillers_do_not_trigger_llm():
    assessment, metadata, tokens = _assess("کرم دست میخوام لطفا نشون بده")
    assert metadata.category == "کرم دست"
    assert assessment.needs_llm is False
    assert "کردن" not in tokens
    assert "دارید" not in tokens


def test_unrelated_leftover_words_are_not_brand_near_misses():
    assert possible_unknown_brand_alias(["تعریق"], None) is None
    assert possible_unknown_brand_alias(["کوین"], None) is None
    assert possible_unknown_brand_alias(["نایس"], None) is None
    assert possible_unknown_brand_alias(["ویتامین"], None) is None


def test_defacto_spelling_is_ambiguous_brand():
    assessment, metadata, tokens = _assess("دیفاکتو کرم دست")
    assert metadata.brand is None
    assert metadata.category == "کرم دست"
    assert possible_unknown_brand_alias(tokens, metadata.brand) == "دیفاکتو"
    assert UNKNOWN_BRAND_ALIAS in assessment.reasons
    assert assessment.needs_llm is True
    assert "category" in assessment.locked_fields


def test_colloquial_moisturizer_is_low_confidence():
    assessment, metadata, _ = _assess("برای مرطوب کردن دست چی دارید؟")
    assert metadata.category is None
    assert assessment.needs_llm is True
    assert IMPLICIT_USE_NO_CATEGORY in assessment.reasons or COLLOQUIAL_NO_METADATA in assessment.reasons


def test_hair_color_mask_is_ambiguous_category():
    assessment, metadata, _ = _assess("ماسک تثبیت رنگ مو")
    assert metadata.category is None
    assert assessment.needs_llm is True
    assert AMBIGUOUS_CATEGORY in assessment.reasons or PARTIAL_PRODUCT_PHRASE in assessment.reasons


def test_hand_and_nail_cream_is_ambiguous_category():
    assessment, metadata, _ = _assess("کرم دست و ناخن")
    assert metadata.category is None
    assert assessment.needs_llm is True
    assert AMBIGUOUS_CATEGORY in assessment.reasons


def test_high_confidence_path_does_not_call_llm():
    llm = ScriptedLlm([{"brand": "Merida", "confidence": 0.99}])
    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    for query in ("رنگ مو", "کرم دست", "لاک ناخن مریدا", "محصولات سالوته"):
        outcome = service.search_with_debug(query)
        assert outcome.debug.llm_metadata_fallback_used is False
    assert llm.calls == 0


def test_fallback_disabled_never_calls_llm():
    llm = ScriptedLlm([{"brand": "Defacto", "category_or_type": "کرم دست", "confidence": 0.9}])
    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config(metadata_fallback_enabled=False)),
        llm_config=_config(metadata_fallback_enabled=False),
    )
    service.search_products("دیفاکتو کرم دست")
    assert llm.calls == 0


def test_fallback_maps_near_miss_brand_after_validation():
    llm = ScriptedLlm(
        [
            {
                "brand": "Defacto",
                "category_or_type": "کرم دست",
                "color": None,
                "material": None,
                "gender": None,
                "price_min": None,
                "price_max": None,
                "confidence": 0.91,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    outcome = service.search_with_debug("دیفاکتو کرم دست")
    assert llm.calls == 1
    assert outcome.debug.llm_metadata_fallback_used is True
    assert repo.last_call["filters"].brand == "Defacto"
    assert repo.last_call["filters"].category == "کرم دست"


def test_hallucinated_brand_is_rejected():
    llm = ScriptedLlm(
        [
            {
                "brand": "Nike",
                "category_or_type": "کرم دست",
                "confidence": 0.99,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("دیفاکتو کرم دست")
    assert repo.last_call["filters"].brand is None
    assert repo.last_call["filters"].category == "کرم دست"


def test_invalid_json_falls_back_to_deterministic():
    llm = ScriptedLlm([ValueError("bad json")])
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    outcome = service.search_with_debug("دیفاکتو کرم دست")
    assert outcome.debug.fallback_reason == "llm_metadata_error"
    assert repo.last_call["filters"].brand is None
    assert repo.last_call["filters"].category == "کرم دست"


def test_timeout_falls_back_to_deterministic():
    llm = ScriptedLlm([TimeoutError("timed out")])
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    outcome = service.search_with_debug("برای مرطوب کردن دست چی دارید؟")
    assert outcome.debug.fallback_reason == "llm_metadata_timeout"
    assert repo.last_call["filters"] == extract_product_metadata("برای مرطوب کردن دست چی دارید؟")


def test_low_llm_confidence_does_not_add_hard_filter():
    merged = merge_llm_understanding(
        extract_product_metadata("برای مرطوب کردن دست چی دارید؟"),
        {
            "brand": None,
            "category_or_type": "کرم دست",
            "confidence": 0.2,
        },
        "برای مرطوب کردن دست چی دارید؟",
        0.7,
    )
    assert merged.metadata.category is None


def test_llm_cannot_reintroduce_suppressed_hand_cream_category():
    llm = ScriptedLlm(
        [
            {
                "brand": None,
                "category_or_type": "کرم دست",
                "confidence": 0.99,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("کرم دست و ناخن")
    assert repo.last_call["filters"].category is None


def test_llm_may_set_different_category_when_hair_dye_was_suppressed():
    llm = ScriptedLlm(
        [
            {
                "brand": None,
                "category_or_type": "ماسک مو",
                "confidence": 0.91,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("ماسک تثبیت رنگ مو")
    assert repo.last_call["filters"].category == "ماسک مو"


def test_unavailable_llm_degrades_to_deterministic():
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=None, config=_config()),
        llm_config=_config(),
    )
    outcome = service.search_with_debug("دیفاکتو کرم دست")
    assert outcome.debug.fallback_reason == "llm_unavailable"
    assert outcome.debug.llm_metadata_fallback_used is False
    assert repo.last_call["filters"].brand is None
    assert repo.last_call["filters"].category == "کرم دست"
    assert outcome.items


def test_missing_ranker_degrades_when_fallback_needed():
    repo = FakeRepository()
    service = ProductService(repo, llm_config=_config())
    outcome = service.search_with_debug("دیفاکتو کرم دست")
    assert outcome.debug.fallback_reason == "llm_unavailable"
    assert outcome.debug.llm_metadata_fallback_used is False
    assert repo.last_call["filters"].category == "کرم دست"


def test_search_emits_structured_product_search_log(caplog):
    import json
    import logging

    service = ProductService(
        FakeRepository(),
        llm_ranker=ProductLlmRanker(generate_json=None, config=_config()),
        llm_config=_config(),
    )
    with caplog.at_level(logging.INFO, logger="app.observability"):
        service.search_with_debug("دیفاکتو کرم دست")
    records = [json.loads(record.getMessage()) for record in caplog.records]
    assert records
    payload = records[-1]
    assert payload["event"] == "product_search_completed"
    assert payload["metadata_source"] == "fallback_failed"
    assert "metadata_confidence" in payload
    assert payload["llm_fallback_triggered"] is True
    assert payload["llm_failure"] is True
    assert payload["fallback_reason"] == "llm_unavailable"
    assert "latency_ms" in payload
    assert "candidate_count" in payload
    assert "result_count" in payload
    assert "query_hash" in payload
    assert "query" not in payload


def test_llm_cannot_remove_locked_deterministic_category():
    llm = ScriptedLlm(
        [
            {
                "brand": "Defacto",
                "category_or_type": "رنگ مو",
                "confidence": 0.99,
            }
        ]
    )
    repo = FakeRepository()
    service = ProductService(
        repo,
        llm_ranker=ProductLlmRanker(generate_json=llm, config=_config()),
        llm_config=_config(),
    )
    service.search_products("دیفاکتو کرم دست")
    assert repo.last_call["filters"].category == "کرم دست"
    assert repo.last_call["filters"].brand == "Defacto"
