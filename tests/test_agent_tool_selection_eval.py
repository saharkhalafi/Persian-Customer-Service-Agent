import pytest

from evaluation.agent_tool_selection_eval import (
    DATASET_PATH,
    evaluate_offline,
    load_dataset,
)
from evaluation.agent_tool_selection_metrics import (
    compare_tool_sets,
    looks_like_clarification,
    search_products_argument_issues,
    score_case,
    verdict_from_metrics,
)


_SKIP_MISSING_DATASET = pytest.mark.skipif(
    not DATASET_PATH.exists(),
    reason="evaluation dataset is local-only and not present in CI",
)


@_SKIP_MISSING_DATASET
def test_dataset_covers_required_intents():
    dataset = load_dataset()
    assert 40 <= len(dataset) <= 60

    by_category = {}
    for item in dataset:
        by_category.setdefault(item["category"], 0)
        by_category[item["category"]] += 1

    for category in (
        "product",
        "order",
        "knowledge",
        "no_tool",
        "multi_tool",
        "clarification",
        "security",
    ):
        assert by_category[category] >= 4

    product_subs = {
        item["subcategory"]
        for item in dataset
        if item["category"] == "product"
    }
    assert {
        "direct",
        "brand_category",
        "attributes",
        "price",
        "colloquial",
        "ambiguous_product",
    } <= product_subs
    assert DATASET_PATH.exists()


def test_set_comparison_tracks_extra_and_missing():
    result = compare_tool_sets(
        ["search_products", "search_knowledge_base"],
        ["search_products", "get_latest_order"],
    )
    assert result["missing_tools"] == ["search_knowledge_base"]
    assert result["extra_tools"] == ["get_latest_order"]
    assert result["set_match"] is False

    optional = compare_tool_sets(
        ["get_latest_order"],
        ["get_latest_order", "get_order_status"],
        optional=["get_order_status"],
    )
    assert optional["set_match"] is True


def test_search_products_rejects_sql_filters_and_customer_id():
    issues = search_products_argument_issues(
        {
            "query": "SELECT * FROM products WHERE brand='x'",
            "customer_id": "9999",
            "filters": {"brand": "x"},
        },
        expected_substrings=["سالوته"],
    )
    assert "forbidden_keys:customer_id,filters" in issues
    assert "sql_in_query" in issues
    assert "missing_substring:سالوته" in issues

    clean = search_products_argument_issues(
        {"query": "محصولات سالوته رو میخوام"},
        expected_substrings=["سالوته"],
    )
    assert clean == []
    assert search_products_argument_issues(
        {"query": "ماساژور شیائومی"},
        expected_substrings=["xiaomi"],
    ) == []


def test_score_case_product_and_security():
    item = {
        "expected_tools": ["search_products"],
        "expected_behavior": "call_tools",
        "expected_query_substrings": ["شیائومی"],
    }
    passed = score_case(
        item,
        ["search_products"],
        [{"name": "search_products", "arguments": {"query": "ماساژور شیائومی"}}],
        expected_customer_id="9206288",
        scoped_customer_ids=["9206288"],
    )
    assert passed["pass"] is True
    assert passed["argument_pass"] is True

    failed = score_case(
        item,
        ["search_products"],
        [
            {
                "name": "search_products",
                "arguments": {
                    "query": "ماساژور شیائومی",
                    "customer_id": "9999",
                    "sql": "select 1",
                },
            }
        ],
    )
    assert failed["pass"] is False
    assert failed["security_violations"]


def test_clarification_and_no_tool_scoring():
    assert looks_like_clarification("کدوم محصول را می‌خواهید؟")

    clarification = score_case(
        {
            "expected_tools": [],
            "expected_behavior": "ask_clarification",
        },
        [],
        [],
        answer="منظورتان کدام سفارشتان است؟",
    )
    assert clarification["clarification_pass"] is True
    assert clarification["pass"] is True

    guessed = score_case(
        {
            "expected_tools": [],
            "expected_behavior": "ask_clarification",
        },
        ["search_products"],
        [{"name": "search_products", "arguments": {"query": "اینو"}}],
        answer="این محصول را پیدا کردم",
    )
    assert guessed["clarification_pass"] is False
    assert "extra=search_products" in guessed["failure_reason"]

    no_tool = score_case(
        {"expected_tools": [], "expected_behavior": "no_tool"},
        [],
        [],
        answer="سلام",
    )
    assert no_tool["no_tool_pass"] is True


@_SKIP_MISSING_DATASET
def test_offline_aggregation_and_verdict():
    dataset = load_dataset()
    predictions = []
    for item in dataset:
        calls = [
            {
                "name": name,
                "arguments": {"query": item["question"]}
                if name == "search_products"
                else {},
            }
            for name in item.get("expected_tools") or []
        ]
        answer = "کدوم مورد را می‌خواهید؟" if item.get("expected_behavior") == "ask_clarification" else "ok"
        predictions.append(
            {
                "id": item["id"],
                "actual_tools": list(item.get("expected_tools") or []),
                "actual_calls": calls,
                "answer": answer,
                "scoped_customer_ids": ["9206288"],
                "expected_customer_id": "9206288",
            }
        )

    payload = evaluate_offline(dataset, predictions)
    assert payload["metrics"]["overall"]["tool_selection_accuracy"] == 100
    assert payload["metrics"]["overall"]["security_accuracy"] == 100
    assert payload["metrics"]["multi_tool"]["exact_match"] == payload["metrics"]["multi_tool"]["total"]
    assert payload["verdict"] == "READY"
    assert payload["failed_cases"] == []


def test_verdict_needs_prompt_fix_when_selection_is_weak():
    metrics = {
        "overall": {
            "tool_selection_accuracy": 62,
            "no_tool_accuracy": 80,
            "security_accuracy": 100,
            "search_products_argument_accuracy": 90,
            "clarification_accuracy": 70,
            "request_context_accuracy": 100,
        },
        "security": {"violations": 0},
        "multi_tool": {"exact_match_accuracy": 40},
        "per_tool": {"search_products": {"recall": 0.8, "fn": 2}},
    }
    assert verdict_from_metrics(metrics) == "NEEDS PROMPT FIX"


def test_verdict_needs_architecture_fix_on_security():
    metrics = {
        "overall": {
            "tool_selection_accuracy": 90,
            "no_tool_accuracy": 90,
            "security_accuracy": 80,
            "search_products_argument_accuracy": 90,
            "clarification_accuracy": 80,
            "request_context_accuracy": 100,
        },
        "security": {"violations": 2},
        "multi_tool": {"exact_match_accuracy": 70},
        "per_tool": {"search_products": {"recall": 0.9, "fn": 1}},
    }
    assert verdict_from_metrics(metrics) == "NEEDS ARCHITECTURE FIX"
