from evaluation.product_search_eval import (
    brands_equivalent,
    family_hit_at_k,
    filter_compliance_at_k,
    hit_at_k,
    infer_evaluation_type,
    item_matches_filters,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_infer_brand_query_is_broad():
    item = {
        "query_type": "BRAND",
        "expected_product_codes": ["1", "2"],
    }
    assert infer_evaluation_type(item) == "broad"


def test_infer_small_direct_product_is_narrow():
    item = {
        "query_type": "DIRECT_PRODUCT",
        "expected_product_codes": ["1118312", "1118368"],
    }
    assert infer_evaluation_type(item) == "narrow"


def test_infer_large_gold_is_broad():
    item = {
        "query_type": "DIRECT_PRODUCT",
        "expected_product_codes": [str(i) for i in range(32)],
    }
    assert infer_evaluation_type(item) == "broad"


def test_unjudged_not_treated_in_precision_denominator():
    retrieved = ["gold", "unjudged", "other"]
    judged = {"gold": 3.0}
    assert precision_at_k(retrieved, judged, 3) == 1.0


def test_ndcg_uses_only_judged_items():
    judged = {"a": 3.0, "b": 1.0}
    retrieved = ["unjudged", "a", "b"]
    score = ndcg_at_k(retrieved, judged, 3)
    assert score is not None
    assert score > 0.9


def test_recall_skipped_logic_via_none_for_empty_relevant():
    assert recall_at_k(["1"], set(), 10) is None


def test_hit_and_filter_compliance():
    assert hit_at_k(["a", "b"], {"b"}, 2) == 1.0
    items = [
        {"brand": "Salute", "sum_of_price": "1000"},
        {"brand": "Note", "sum_of_price": "1000"},
    ]
    assert filter_compliance_at_k(items, {"brand": "Salute"}, 2) == 0.5


def test_price_filter_uses_sum_of_price():
    item = {"brand": "Note", "sum_of_price": "250000.0"}
    assert item_matches_filters(item, {"brand": "Note", "max_price": 300000})
    assert not item_matches_filters(item, {"brand": "Note", "max_price": 200000})


def test_family_hit_uses_brand_and_type_not_sku():
    gold = [{"brand": "Merida", "type": "لاک ناخن", "product_code": "869553"}]
    retrieved = [
        {"brand": "Merida", "type": "لاک ناخن", "product_code": "866539"},
        {"brand": "Salute", "type": "لاک ناخن", "product_code": "1"},
    ]
    assert family_hit_at_k(retrieved, gold, 1) == 1.0
    assert family_hit_at_k(retrieved[1:], gold, 1) == 0.0


def test_brand_aliases_match():
    assert brands_equivalent("سالوته", "Salute")
    assert brands_equivalent("xiaomi", "Xiaomi")
    assert brands_equivalent("نوت", "Note")
