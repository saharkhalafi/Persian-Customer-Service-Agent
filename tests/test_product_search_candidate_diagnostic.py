from evaluation.product_search_candidate_diagnostic import (
    classify_narrow_case,
    extracted_filter_exclusions,
)


def test_ranking_failure_when_gold_is_in_filtered_set():
    classification, evidence = classify_narrow_case(
        gold_codes=["869553"],
        catalog={
            "869553": {
                "product_code": "869553",
                "brand": "Merida",
                "type": "لاک ناخن",
                "product_name": "لاک ناخن مریدا 123",
                "sum_of_price": "100000",
                "last_status": "Enable",
            }
        },
        expected_filters={"brand": "Merida", "category_or_type": "لاک ناخن"},
        metadata={"brand": "Merida", "category": "لاک ناخن"},
        gold_in_filtered=["869553"],
        best_rank=47,
        token_filters=[],
        apply_token_filters=False,
    )
    assert classification == "RANKING_FAILURE"
    assert any("47" in item for item in evidence)


def test_filter_failure_when_extracted_category_excludes_gold():
    gold = {
        "product_code": "827608",
        "brand": "X",
        "type": "ماسک مو",
        "category_level1": "مو",
        "category_level2": "ماسک",
        "product_name": "ماسک تثبیت کننده رنگ مو",
        "sum_of_price": "100000",
        "last_status": "Enable",
    }
    assert "category" in extracted_filter_exclusions(gold, {"category": "رنگ مو"})
    classification, evidence = classify_narrow_case(
        gold_codes=["827608"],
        catalog={"827608": gold},
        expected_filters={"product_name_contains": "ماسک تثبیت کننده رنگ مو"},
        metadata={"category": "رنگ مو"},
        gold_in_filtered=[],
        best_rank=None,
        token_filters=[],
        apply_token_filters=False,
    )
    assert classification == "FILTER_FAILURE"
    assert any("category" in item for item in evidence)


def test_candidate_generation_failure_when_token_filters_drop_gold():
    gold = {
        "product_code": "1",
        "brand": "Salute",
        "type": "کرم دست",
        "category_level1": "کرم",
        "category_level2": "",
        "product_name": "کرم دست سالوته",
        "sum_of_price": "80000",
        "last_status": "Enable",
    }
    classification, _evidence = classify_narrow_case(
        gold_codes=["1"],
        catalog={"1": gold},
        expected_filters={"brand": "Salute"},
        metadata={"brand": "Salute"},
        gold_in_filtered=[],
        best_rank=None,
        token_filters=["حاوی", "ویتامین"],
        apply_token_filters=True,
    )
    assert classification == "CANDIDATE_GENERATION_FAILURE"


def test_no_catalog_match():
    classification, evidence = classify_narrow_case(
        gold_codes=["missing"],
        catalog={},
        expected_filters={},
        metadata={},
        gold_in_filtered=[],
        best_rank=None,
        token_filters=[],
        apply_token_filters=False,
    )
    assert classification == "NO_CATALOG_MATCH"
    assert evidence


def test_gold_issue_when_catalog_row_fails_expected_filters():
    classification, _evidence = classify_narrow_case(
        gold_codes=["1"],
        catalog={"1": {"brand": "Note", "type": "ماسک مو", "product_name": "x"}},
        expected_filters={"brand": "Merida", "category_or_type": "لاک ناخن"},
        metadata={"brand": "Merida"},
        gold_in_filtered=[],
        best_rank=None,
        token_filters=[],
        apply_token_filters=False,
    )
    assert classification == "DATA/GOLD_ISSUE"


def test_success_is_not_classified():
    classification, _evidence = classify_narrow_case(
        gold_codes=["1"],
        catalog={"1": {"brand": "Rodier", "type": "عطر", "product_name": "عطر زنانه رودیر"}},
        expected_filters={"brand": "RODIER"},
        metadata={"brand": "RODIER", "category": "عطر"},
        gold_in_filtered=["1"],
        best_rank=1,
        token_filters=[],
        apply_token_filters=False,
    )
    assert classification is None
