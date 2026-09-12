"""Product search benchmark on the real search_products pipeline.

Does not mock retrieval. Does not change product-search behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.core.database import SessionLocal
from app.repositories.product_repository import ProductRepository
from app.services.product_brand_aliases import canonicalize_brand
from app.services.product_llm_config import ProductLlmConfig
from app.services.product_llm_ranker import ProductLlmRanker, gemini_generate_json
from app.services.product_service import ProductService


DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "Eval_product_search.json"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "product_search_benchmark.json"
COMPARISON_PATH = PROJECT_ROOT / "evaluation" / "results" / "product_search_llm_comparison.json"
LIMIT = 10
KS = (3, 5, 10)
BROAD_GOLD_THRESHOLD = 20
LLM_INPUT_USD_PER_MILLION = 0.30
LLM_OUTPUT_USD_PER_MILLION = 2.50

PIPELINE_CONFIGS = {
    "baseline": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=False,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": False,
        "negative_constraints_enabled": False,
    },
    "production": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=False,
            metadata_fallback_enabled=True,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": False,
        "negative_constraints_enabled": True,
    },
    "metadata_fallback": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=False,
            metadata_fallback_enabled=True,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": False,
        "negative_constraints_enabled": True,
    },
    "name_ranking": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=False,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": True,
        "negative_constraints_enabled": True,
    },
    "query_understanding": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=True,
            rerank_enabled=False,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": True,
        "negative_constraints_enabled": True,
    },
    "rerank": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=True,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": True,
        "negative_constraints_enabled": True,
    },
    "full": {
        "llm": ProductLlmConfig(
            query_understanding_enabled=True,
            rerank_enabled=True,
            candidate_k=20,
            final_k=10,
        ),
        "name_matching_enabled": True,
        "negative_constraints_enabled": True,
    },
}

BROAD_TYPES = {
    "BRAND",
    "CATEGORY",
    "GENDER_FILTER",
    "PRICE_FILTER",
    "AVAILABILITY_FILTER",
    "MULTI_ATTRIBUTE",
}
NARROW_TYPES = {
    "DIRECT_PRODUCT",
    "BRAND_CATEGORY",
    "ATTRIBUTE",
    "PARTIAL_PRODUCT_NAME",
    "COLLOQUIAL",
    "PERSIAN_VARIATION",
}
SPECIAL_TYPES = {
    "AMBIGUOUS",
    "NO_RESULT",
    "OUT_OF_SCOPE",
    "NEGATIVE_CONSTRAINT",
}


def _codes(item: dict) -> list[str]:
    return [str(code) for code in (item.get("expected_product_codes") or []) if str(code).strip()]


def _relevance(item: dict) -> dict[str, float]:
    raw = item.get("relevance") or {}
    judged: dict[str, float] = {}
    for code, grade in raw.items():
        try:
            value = float(grade)
        except (TypeError, ValueError):
            continue
        if value > 0:
            judged[str(code)] = value
    if judged:
        return judged
    return {code: 3.0 for code in _codes(item)}


def infer_evaluation_type(item: dict) -> str:
    if item.get("evaluation_type") in {"narrow", "broad", "special"}:
        return item["evaluation_type"]
    query_type = item.get("query_type") or ""
    gold = _codes(item)
    if query_type in SPECIAL_TYPES:
        return "special"
    if query_type in BROAD_TYPES or len(gold) >= BROAD_GOLD_THRESHOLD:
        return "broad"
    if query_type in NARROW_TYPES:
        return "narrow"
    return "broad" if gold else "special"


def annotate_item(item: dict) -> dict:
    updated = dict(item)
    query = updated.get("query") or ""
    query_type = updated.get("query_type")
    filters = dict(updated.get("expected_filters") or {})
    gold = _codes(updated)
    notes: list[str] = []

    if updated.get("id") == "product_eval_001":
        if filters.get("brand") == "Troya" or filters.get("category_or_type") == "لاک ناخن":
            filters = {"category_or_type": "ماسک مو"}
            notes.append("original filters Troya/لاک ناخن contradicted the hair-mask query")
    if updated.get("id") == "product_eval_011":
        if filters.get("brand") == "Merida" or "Signature" in query:
            filters = {"brand": "Signature", "category_or_type": "پنکک"}
            notes.append("original brand filter Merida contradicted Signature pancake query; gold may be invalid")

    eval_type = infer_evaluation_type({**updated, "expected_product_codes": gold})
    if query_type == "BRAND" or query_type == "CATEGORY":
        eval_type = "broad"
    if query_type in SPECIAL_TYPES:
        eval_type = "special"
    if query_type in NARROW_TYPES and len(gold) >= BROAD_GOLD_THRESHOLD:
        eval_type = "broad"
        notes.append("large gold set treated as judged relevance, not exhaustive recall")

    rel = updated.get("relevance") or {}
    if rel and set(map(str, rel)) != set(gold) and gold:
        notes.append("relevance keys and expected_product_codes differ; relevance used as judged set")

    updated["expected_filters"] = filters
    updated["evaluation_type"] = eval_type
    updated["needs_review"] = bool(notes) or bool(updated.get("needs_review"))
    if notes:
        updated["review_notes"] = "; ".join(notes)
    return updated


def annotate_dataset(path: Path = DATASET_PATH) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    updated = [annotate_item(item) for item in data]
    path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    text_value = re.sub(r"[^0-9.]+", "", str(value))
    if not text_value:
        return None
    try:
        return float(text_value)
    except ValueError:
        return None


def brands_equivalent(actual: str | None, expected: str | None) -> bool:
    if not actual or not expected:
        return False
    if actual.casefold().strip() == expected.casefold().strip():
        return True
    left = canonicalize_brand(actual) or actual
    right = canonicalize_brand(expected) or expected
    return left.casefold() == right.casefold()


def category_matches(item: dict, expected: str) -> bool:
    expected_l = expected.casefold()
    fields = [
        item.get("type"),
        item.get("category_level1"),
        item.get("category_level2"),
        item.get("product_name"),
    ]
    return any(expected_l in str(field or "").casefold() for field in fields)


def gender_matches(actual: str | None, expected: str | None) -> bool:
    if not expected:
        return True
    if not actual:
        return False
    return expected.casefold() in actual.casefold()


def item_matches_filters(item: dict, filters: dict) -> bool:
    if not filters:
        return True
    brand = filters.get("brand")
    if brand and not brands_equivalent(item.get("brand"), brand):
        return False
    category = filters.get("category") or filters.get("category_or_type")
    if category and not category_matches(item, str(category)):
        return False
    color = filters.get("color")
    if color and color.casefold() not in str(item.get("color") or "").casefold():
        return False
    gender = filters.get("gender") or filters.get("gender_contains")
    if gender and not gender_matches(item.get("gender"), str(gender)):
        return False
    price_max = filters.get("price_max", filters.get("max_price"))
    if price_max is not None:
        price = parse_price(item.get("sum_of_price"))
        if price is None or price > float(price_max):
            return False
    price_min = filters.get("price_min", filters.get("min_price"))
    if price_min is not None:
        price = parse_price(item.get("sum_of_price"))
        if price is None or price < float(price_min):
            return False
    if filters.get("available") is True or filters.get("availability") is True:
        if str(item.get("last_status") or "") != "Enable":
            return False
    exclude = filters.get("exclude_brand")
    if exclude and brands_equivalent(item.get("brand"), str(exclude)):
        return False
    contains = filters.get("product_name_contains")
    if contains and contains.casefold() not in str(item.get("product_name") or "").casefold():
        return False
    return True


def filter_compliance_at_k(items: list[dict], filters: dict, k: int) -> float | None:
    if not filters:
        return None
    top = items[:k]
    if not top:
        return 0.0
    matched = sum(1 for item in top if item_matches_filters(item, filters))
    return matched / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(code in relevant for code in retrieved[:k]) else 0.0


def mrr_score(retrieved: list[str], relevant: set[str]) -> float:
    for rank, code in enumerate(retrieved, start=1):
        if code in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved: list[str], judged: dict[str, float], k: int) -> float | None:
    top = retrieved[:k]
    judged_in_top = [code for code in top if code in judged]
    if not judged_in_top:
        if any(judged.values()):
            return 0.0 if top else None
        return None
    relevant_in_top = [code for code in judged_in_top if judged[code] > 0]
    return len(relevant_in_top) / len(judged_in_top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    return sum(1 for code in retrieved[:k] if code in relevant) / len(relevant)


def dcg_at_k(grades: list[float], k: int) -> float:
    score = 0.0
    for index, grade in enumerate(grades[:k], start=1):
        score += (2**grade - 1) / math.log2(index + 1)
    return score


def ndcg_at_k(retrieved: list[str], judged: dict[str, float], k: int) -> float | None:
    condensed = [judged[code] for code in retrieved if code in judged]
    if not judged or not condensed:
        return None
    ideal = sorted(judged.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, min(k, len(ideal)))
    if ideal_dcg == 0:
        return None
    return dcg_at_k(condensed, min(k, len(condensed))) / ideal_dcg


def metadata_field_accuracy(extracted: dict, filters: dict) -> tuple[int, int]:
    checked = 0
    correct = 0
    if filters.get("brand"):
        checked += 1
        if brands_equivalent(extracted.get("brand"), filters.get("brand")):
            correct += 1
    category = filters.get("category") or filters.get("category_or_type")
    if category:
        checked += 1
        actual = str(extracted.get("category") or "")
        if actual and (
            category.casefold() in actual.casefold()
            or actual.casefold() in category.casefold()
        ):
            correct += 1
    if filters.get("gender") or filters.get("gender_contains"):
        checked += 1
        expected = filters.get("gender") or filters.get("gender_contains")
        actual = extracted.get("gender")
        if actual and expected and str(expected).casefold() in str(actual).casefold():
            correct += 1
    if filters.get("color"):
        checked += 1
        if str(extracted.get("color") or "").casefold() == str(filters.get("color")).casefold():
            correct += 1
    if filters.get("max_price") is not None or filters.get("price_max") is not None:
        checked += 1
        expected = float(filters.get("price_max", filters.get("max_price")))
        actual = extracted.get("price_max")
        if actual is not None and abs(float(actual) - expected) < 1:
            correct += 1
    if filters.get("min_price") is not None or filters.get("price_min") is not None:
        checked += 1
        expected = float(filters.get("price_min", filters.get("min_price")))
        actual = extracted.get("price_min")
        if actual is not None and abs(float(actual) - expected) < 1:
            correct += 1
    if filters.get("available") is True or filters.get("availability") is True:
        checked += 1
        if extracted.get("availability") is True:
            correct += 1
    if filters.get("exclude_brand"):
        checked += 1
        excluded = extracted.get("exclude_brands") or []
        if any(brands_equivalent(item, str(filters.get("exclude_brand"))) for item in excluded):
            correct += 1
    return correct, checked


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} ({value * 100:.2f}%)"


def classify_failure(item: dict, case: dict) -> str | None:
    if case.get("pass"):
        return None
    query_type = item.get("query_type")
    filters = item.get("expected_filters") or {}
    retrieved = case.get("retrieved_codes") or []
    judged = set((_relevance(item) or {}).keys())
    eval_type = case.get("evaluation_type")

    if item.get("id") == "product_eval_011":
        return "OTHER"
    if query_type == "NEGATIVE_CONSTRAINT":
        return "NEGATIVE_CONSTRAINT"
    if query_type == "PERSIAN_VARIATION" and any(
        token in (item.get("query") or "") for token in ("تریا", "دیفاکتو")
    ):
        return "TYPO_VARIATION"
    if query_type in {"NO_RESULT", "OUT_OF_SCOPE"}:
        return "NO_CATALOG_MATCH" if retrieved else "NO_RESULT"
    if case.get("metadata_correct") is False:
        if case.get("llm_query_understanding_used"):
            return "QUERY_UNDERSTANDING_MISS"
        return "METADATA_MISS"
    if filters and (case.get("filter_compliance@10") or 0) < 1:
        return "FILTER_MISS"
    if not retrieved:
        return "NO_RESULT" if query_type == "NO_RESULT" else "NO_CATALOG_MATCH"
    hits = [code for code in retrieved if code in judged]
    if hits:
        return "RANKING_MISS"
    if eval_type == "narrow" or judged:
        if case.get("llm_rerank_used"):
            return "LLM_RANKING_MISS"
        return "RANKING_MISS"
    if filters:
        return "METADATA_MISS"
    return "OTHER"


def family_keys(item: dict) -> set[tuple[str, str]]:
    brand = (canonicalize_brand(item.get("brand")) or item.get("brand") or "").casefold().strip()
    product_type = str(item.get("type") or "").casefold().strip()
    if not brand or not product_type:
        return set()
    return {(brand, product_type)}


def family_hit_at_k(
    retrieved_items: list[dict],
    gold_items: list[dict],
    k: int,
) -> float | None:
    families: set[tuple[str, str]] = set()
    for item in gold_items:
        families.update(family_keys(item))
    if not families:
        return None
    for item in retrieved_items[:k]:
        if family_keys(item) & families:
            return 1.0
    return 0.0


def attach_db_facts(db, items: list[dict]) -> list[dict]:
    codes = [item.get("product_code") for item in items if item.get("product_code")]
    if not codes:
        return items
    rows = db.execute(
        text(
            """
            SELECT
                product_code, brand, type, category_level1, category_level2,
                gender, color, sum_of_price, special_price, last_status,
                product_name
            FROM public.products
            WHERE product_code = ANY(:codes)
            """
        ),
        {"codes": codes},
    ).mappings()
    by_code = {str(row["product_code"]): dict(row) for row in rows}
    enriched = []
    for item in items:
        extra = by_code.get(str(item.get("product_code")) or "", {})
        merged = {**item, **extra}
        enriched.append(merged)
    return enriched


def evaluate_case(service: ProductService, db, item: dict) -> dict:
    query = item["query"]
    eval_type = infer_evaluation_type(item)
    filters = item.get("expected_filters") or {}
    judged = _relevance(item)
    relevant = {code for code, grade in judged.items() if grade > 0}
    exhaustive = eval_type == "narrow" and bool(relevant) and len(relevant) < BROAD_GOLD_THRESHOLD

    outcome = service.search_with_debug(query=query, limit=LIMIT)
    extracted = outcome.debug.metadata
    results = outcome.items
    retrieved_items = [
        {
            "product_code": item_row.product_code,
            "product_name": item_row.product_name,
            "brand": item_row.brand,
            "type": item_row.type,
            "color": item_row.color,
            "last_status": item_row.last_status,
        }
        for item_row in results
    ]
    retrieved_items = attach_db_facts(db, retrieved_items)
    retrieved_codes = [str(row["product_code"]) for row in retrieved_items if row.get("product_code")]
    gold_items = attach_db_facts(
        db,
        [{"product_code": code} for code in relevant],
    )

    meta_correct, meta_checked = metadata_field_accuracy(extracted, filters)
    compliance = {
        f"filter_compliance@{k}": filter_compliance_at_k(retrieved_items, filters, k)
        for k in KS
    }

    ranking: dict[str, float | None] = {}
    if relevant and eval_type != "special":
        ranking["hit@3"] = hit_at_k(retrieved_codes, relevant, 3)
        ranking["hit@5"] = hit_at_k(retrieved_codes, relevant, 5)
        ranking["hit@10"] = hit_at_k(retrieved_codes, relevant, 10)
        ranking["mrr"] = mrr_score(retrieved_codes, relevant)
        ranking["precision@3"] = precision_at_k(retrieved_codes, judged, 3)
        ranking["precision@5"] = precision_at_k(retrieved_codes, judged, 5)
        ranking["precision@10"] = precision_at_k(retrieved_codes, judged, 10)
        ranking["ndcg@3"] = ndcg_at_k(retrieved_codes, judged, 3)
        ranking["ndcg@5"] = ndcg_at_k(retrieved_codes, judged, 5)
        ranking["ndcg@10"] = ndcg_at_k(retrieved_codes, judged, 10)
        if exhaustive:
            ranking["recall@3"] = recall_at_k(retrieved_codes, relevant, 3)
            ranking["recall@5"] = recall_at_k(retrieved_codes, relevant, 5)
            ranking["recall@10"] = recall_at_k(retrieved_codes, relevant, 10)
        ranking["family_hit@3"] = family_hit_at_k(retrieved_items, gold_items, 3)
        ranking["family_hit@5"] = family_hit_at_k(retrieved_items, gold_items, 5)
        ranking["family_hit@10"] = family_hit_at_k(retrieved_items, gold_items, 10)

    behavior = item.get("expected_behavior")
    passed = True
    if eval_type == "broad" and filters:
        passed = (compliance["filter_compliance@10"] or 0) >= 1.0
    elif eval_type == "narrow" and relevant:
        passed = (ranking.get("hit@10") or 0) >= 1.0
    elif behavior in {"NO_RESULTS", "OUT_OF_SCOPE"}:
        passed = len(retrieved_codes) == 0
    elif behavior == "SEARCH_OR_CLARIFY":
        passed = True
    elif behavior == "RETURN_RESULTS":
        passed = bool(retrieved_codes)

    case = {
        "id": item.get("id"),
        "query": query,
        "query_type": item.get("query_type"),
        "evaluation_type": eval_type,
        "needs_review": bool(item.get("needs_review")),
        "expected_filters": filters,
        "expected_codes": _codes(item),
        "extracted": extracted,
        "deterministic_extracted": outcome.debug.deterministic_metadata,
        "retrieved_codes": retrieved_codes,
        "retrieved_preview": [
            {
                "product_code": row.get("product_code"),
                "brand": row.get("brand"),
                "type": row.get("type"),
                "name": row.get("product_name"),
            }
            for row in retrieved_items[:5]
        ],
        "metadata_correct": None if meta_checked == 0 else meta_correct == meta_checked,
        "metadata_field_hits": meta_correct,
        "metadata_field_checked": meta_checked,
        **compliance,
        **ranking,
        "pass": passed,
        "llm_query_understanding_used": outcome.debug.llm_query_understanding_used,
        "llm_metadata_fallback_used": outcome.debug.llm_metadata_fallback_used,
        "metadata_confidence": outcome.debug.metadata_confidence,
        "metadata_confidence_reasons": outcome.debug.metadata_confidence_reasons,
        "llm_rerank_used": outcome.debug.llm_rerank_used,
        "db_candidate_count": outcome.debug.db_candidate_count,
        "llm_candidate_count": outcome.debug.llm_candidate_count,
        "llm_calls": outcome.debug.llm_calls,
        "llm_input_tokens": outcome.debug.llm_input_tokens,
        "llm_output_tokens": outcome.debug.llm_output_tokens,
        "latency_ms": outcome.debug.latency_ms,
        "fallback_reason": outcome.debug.fallback_reason,
    }
    case["failure_reason"] = classify_failure(item, case)
    return case


def aggregate(cases: list[dict]) -> dict:
    def collect(pred, key):
        return [case.get(key) for case in cases if pred(case)]

    exact = [case for case in cases if case["evaluation_type"] == "narrow"]
    broad = [case for case in cases if case["evaluation_type"] == "broad"]
    with_filters = [case for case in cases if case.get("expected_filters")]
    with_meta = [case for case in cases if case.get("metadata_field_checked")]

    by_category: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        grouped[case.get("query_type") or "unknown"].append(case)
    for name, group in sorted(grouped.items()):
        by_category[name] = {
            "count": len(group),
            "Hit@10": mean([case.get("hit@10") for case in group]),
            "MRR": mean([case.get("mrr") for case in group]),
            "Precision@10": mean([case.get("precision@10") for case in group]),
            "Recall@10": mean([case.get("recall@10") for case in group]),
            "NDCG@10": mean([case.get("ndcg@10") for case in group]),
            "Filter Compliance@10": mean(
                [case.get("filter_compliance@10") for case in group]
            ),
        }

    meta_hits = sum(case.get("metadata_field_hits") or 0 for case in with_meta)
    meta_checked = sum(case.get("metadata_field_checked") or 0 for case in with_meta)
    focus_types = (
        "PARTIAL_PRODUCT_NAME",
        "PERSIAN_VARIATION",
        "COLLOQUIAL",
        "BRAND_CATEGORY",
    )
    focus = {name: by_category.get(name) for name in focus_types if name in by_category}
    metadata_miss_cases = [case for case in cases if case.get("failure_reason") == "METADATA_MISS"]
    focus["METADATA_MISS"] = {
        "count": len(metadata_miss_cases),
        "Hit@10": mean([case.get("hit@10") for case in metadata_miss_cases]),
        "MRR": mean([case.get("mrr") for case in metadata_miss_cases]),
        "Precision@10": mean([case.get("precision@10") for case in metadata_miss_cases]),
        "Recall@10": mean([case.get("recall@10") for case in metadata_miss_cases]),
        "NDCG@10": mean([case.get("ndcg@10") for case in metadata_miss_cases]),
        "Filter Compliance@10": mean(
            [case.get("filter_compliance@10") for case in metadata_miss_cases]
        ),
    }

    return {
        "total_queries": len(cases),
        "narrow_count": len(exact),
        "broad_count": len(broad),
        "special_count": sum(1 for case in cases if case["evaluation_type"] == "special"),
        "exact": {
            "Hit@3": mean(collect(lambda c: c["evaluation_type"] == "narrow", "hit@3")),
            "Hit@5": mean(collect(lambda c: c["evaluation_type"] == "narrow", "hit@5")),
            "Hit@10": mean(collect(lambda c: c["evaluation_type"] == "narrow", "hit@10")),
            "MRR": mean(collect(lambda c: c["evaluation_type"] == "narrow", "mrr")),
            "Precision@3": mean(collect(lambda c: c["evaluation_type"] == "narrow", "precision@3")),
            "Precision@5": mean(collect(lambda c: c["evaluation_type"] == "narrow", "precision@5")),
            "Precision@10": mean(collect(lambda c: c["evaluation_type"] == "narrow", "precision@10")),
            "Recall@3": mean(collect(lambda c: c["evaluation_type"] == "narrow", "recall@3")),
            "Recall@5": mean(collect(lambda c: c["evaluation_type"] == "narrow", "recall@5")),
            "Recall@10": mean(collect(lambda c: c["evaluation_type"] == "narrow", "recall@10")),
            "NDCG@3": mean(collect(lambda c: c["evaluation_type"] == "narrow", "ndcg@3")),
            "NDCG@5": mean(collect(lambda c: c["evaluation_type"] == "narrow", "ndcg@5")),
            "NDCG@10": mean(collect(lambda c: c["evaluation_type"] == "narrow", "ndcg@10")),
            "Family Hit@3": mean(
                collect(lambda c: c["evaluation_type"] == "narrow", "family_hit@3")
            ),
            "Family Hit@5": mean(
                collect(lambda c: c["evaluation_type"] == "narrow", "family_hit@5")
            ),
            "Family Hit@10": mean(
                collect(lambda c: c["evaluation_type"] == "narrow", "family_hit@10")
            ),
        },
        "broad": {
            "Filter Compliance@3": mean(
                collect(lambda c: c["evaluation_type"] == "broad", "filter_compliance@3")
            ),
            "Filter Compliance@5": mean(
                collect(lambda c: c["evaluation_type"] == "broad", "filter_compliance@5")
            ),
            "Filter Compliance@10": mean(
                collect(lambda c: c["evaluation_type"] == "broad", "filter_compliance@10")
            ),
        },
        "metadata_filter_accuracy": (meta_hits / meta_checked) if meta_checked else None,
        "by_category": by_category,
        "needs_review": [
            case["id"]
            for case in cases
            if case.get("needs_review")
        ],
        "filter_queries": len(with_filters),
        "focus": focus,
        "operations": operational_metrics(cases),
        "failure_counts": failure_counts(cases),
        "hard_filter_violations": hard_filter_violations(cases),
    }


def operational_metrics(cases: list[dict]) -> dict:
    latencies = [float(case.get("latency_ms") or 0) for case in cases]
    llm_calls = [int(case.get("llm_calls") or 0) for case in cases]
    input_tokens = [int(case.get("llm_input_tokens") or 0) for case in cases]
    output_tokens = [int(case.get("llm_output_tokens") or 0) for case in cases]
    avg_input = mean(input_tokens) or 0.0
    avg_output = mean(output_tokens) or 0.0
    estimated_cost = (
        (sum(input_tokens) / 1_000_000) * LLM_INPUT_USD_PER_MILLION
        + (sum(output_tokens) / 1_000_000) * LLM_OUTPUT_USD_PER_MILLION
    )
    llm_query_count = sum(1 for count in llm_calls if count > 0)
    total_llm_calls = sum(llm_calls)
    return {
        "total_queries": len(cases),
        "llm_calls": total_llm_calls,
        "llm_call_rate": (llm_query_count / len(cases)) if cases else 0.0,
        "avg_latency_ms": mean(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "llm_calls_per_query": mean(llm_calls),
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "estimated_cost_usd": estimated_cost,
        "llm_failures": sum(1 for case in cases if case.get("fallback_reason")),
        "llm_understanding_queries": sum(
            1 for case in cases if case.get("llm_query_understanding_used")
        ),
        "llm_metadata_fallback_queries": sum(
            1 for case in cases if case.get("llm_metadata_fallback_used")
        ),
        "llm_rerank_queries": sum(1 for case in cases if case.get("llm_rerank_used")),
    }


def failure_counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        reason = case.get("failure_reason")
        if reason:
            counts[reason] += 1
    return dict(sorted(counts.items()))


def hard_filter_violations(cases: list[dict]) -> int:
    violations = 0
    for case in cases:
        filters = case.get("expected_filters") or {}
        if not filters:
            continue
        compliance = case.get("filter_compliance@10")
        if compliance is not None and compliance < 1:
            violations += 1
    return violations


def print_report(summary: dict, failures: list[dict]) -> None:
    print("=== Product Search Benchmark ===")
    print()
    print(f"Total queries: {summary['total_queries']}")
    print(f"Narrow: {summary['narrow_count']}")
    print(f"Broad: {summary['broad_count']}")
    print(f"Special: {summary['special_count']}")
    print()
    print("Exact/Narrow:")
    for key, value in summary["exact"].items():
        print(f"  {key}: {fmt(value)}")
    print()
    print("Broad/Metadata:")
    for key, value in summary["broad"].items():
        print(f"  {key}: {fmt(value)}")
    print()
    print("Metadata:")
    print(f"  Metadata Filter Accuracy: {fmt(summary['metadata_filter_accuracy'])}")
    print()
    print("By Category:")
    for name, stats in summary["by_category"].items():
        print(f"  {name} (n={stats['count']})")
        for key in (
            "Hit@10",
            "MRR",
            "Precision@10",
            "Recall@10",
            "NDCG@10",
            "Filter Compliance@10",
        ):
            print(f"    {key}: {fmt(stats[key])}")
    print()
    print("Focus slices:")
    for name, stats in (summary.get("focus") or {}).items():
        if not stats:
            continue
        print(f"  {name} (n={stats['count']})")
        for key in (
            "Hit@10",
            "MRR",
            "Precision@10",
            "Recall@10",
            "NDCG@10",
            "Filter Compliance@10",
        ):
            print(f"    {key}: {fmt(stats[key])}")
    print()
    print("Operations:")
    ops = summary.get("operations") or {}
    print(f"  Total queries: {ops.get('total_queries')}")
    print(f"  LLM calls: {ops.get('llm_calls')}")
    print(f"  LLM call rate: {ops.get('llm_call_rate')}")
    print(f"  Avg latency: {ops.get('avg_latency_ms')}")
    print(f"  Median latency: {ops.get('median_latency_ms')}")
    print(f"  LLM calls/query: {ops.get('llm_calls_per_query')}")
    print(f"  LLM failures: {ops.get('llm_failures')}")
    print(f"  LLM metadata fallback queries: {ops.get('llm_metadata_fallback_queries')}")
    print(f"  Avg input tokens: {ops.get('avg_input_tokens')}")
    print(f"  Avg output tokens: {ops.get('avg_output_tokens')}")
    print(f"  Estimated cost USD: {ops.get('estimated_cost_usd')}")
    print(f"  Hard-filter violations: {summary.get('hard_filter_violations')}")
    print()
    print("Important failures:")
    for case in failures[:12]:
        print(f"- {case['id']} [{case.get('failure_reason')}] {case['query']}")
        print(f"  expected filters: {case.get('expected_filters')}")
        print(f"  expected codes: {case.get('expected_codes')[:8]}")
        print(f"  retrieved: {case.get('retrieved_codes')}")
        print(f"  extracted: {case.get('extracted')}")


def build_service(db, pipeline: dict, generate_json=None) -> ProductService:
    config = pipeline["llm"]
    ranker = None
    if (
        config.query_understanding_enabled
        or config.rerank_enabled
        or config.metadata_fallback_enabled
    ):
        ranker = ProductLlmRanker(generate_json=generate_json, config=config)
    return ProductService(
        ProductRepository(db),
        llm_ranker=ranker,
        llm_config=config,
        name_matching_enabled=pipeline.get("name_matching_enabled", True),
        negative_constraints_enabled=pipeline.get("negative_constraints_enabled", True),
    )


def comparison_row(name: str, summary: dict) -> dict:
    exact = summary.get("exact") or {}
    broad = summary.get("broad") or {}
    ops = summary.get("operations") or {}
    return {
        "configuration": name,
        "Hit@10": exact.get("Hit@10"),
        "MRR": exact.get("MRR"),
        "NDCG@10": exact.get("NDCG@10"),
        "Filter Compliance@10": broad.get("Filter Compliance@10"),
        "Avg Latency": ops.get("avg_latency_ms"),
        "LLM Calls/Query": ops.get("llm_calls_per_query"),
        "Avg Input Tokens": ops.get("avg_input_tokens"),
        "Avg Output Tokens": ops.get("avg_output_tokens"),
        "Estimated Cost USD": ops.get("estimated_cost_usd"),
    }


def print_comparison_table(rows: list[dict]) -> None:
    print()
    print("=== Configuration Comparison ===")
    header = (
        f"{'Configuration':<24} {'Hit@10':>8} {'MRR':>8} {'NDCG@10':>8} "
        f"{'Filt@10':>8} {'Avg Lat':>10} {'LLM/q':>8}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['configuration']:<24} "
            f"{_cell(row['Hit@10']):>8} "
            f"{_cell(row['MRR']):>8} "
            f"{_cell(row['NDCG@10']):>8} "
            f"{_cell(row['Filter Compliance@10']):>8} "
            f"{_cell(row['Avg Latency'], 1):>10} "
            f"{_cell(row['LLM Calls/Query']):>8}"
        )


def _cell(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def try_gemini_generator():
    if not os.getenv("GEMINI_API_KEY"):
        return None
    try:
        from app.core.gemini import GeminiClient

        return gemini_generate_json(GeminiClient())
    except Exception:
        return None


def run_configuration(name: str, dataset: list[dict], db, generate_json) -> dict:
    config = PIPELINE_CONFIGS[name]
    service = build_service(db, config, generate_json=generate_json)
    cases = []
    for index, item in enumerate(dataset, start=1):
        print(f"[{name} {index}/{len(dataset)}] {item['id']}")
        cases.append(evaluate_case(service, db, item))
    summary = aggregate(cases)
    failures = [case for case in cases if not case.get("pass")]
    failures.sort(key=lambda case: (case.get("failure_reason") or "", case.get("id") or ""))
    return {
        "name": name,
        "config": {
            **config["llm"].as_dict(),
            "name_matching_enabled": config.get("name_matching_enabled"),
            "negative_constraints_enabled": config.get("negative_constraints_enabled"),
        },
        "summary": summary,
        "failures": failures,
        "cases": cases,
        "hard_filter_violations": hard_filter_violations(cases),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="production",
        help="Comma-separated: production,metadata_fallback,baseline,name_ranking,query_understanding,rerank,full",
    )
    args = parser.parse_args()
    selected = [item.strip() for item in args.configs.split(",") if item.strip()]
    unknown = [item for item in selected if item not in PIPELINE_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}")

    dataset = annotate_dataset()
    generate_json = try_gemini_generator()
    needs_llm = any(
        PIPELINE_CONFIGS[item]["llm"].query_understanding_enabled
        or PIPELINE_CONFIGS[item]["llm"].rerank_enabled
        or PIPELINE_CONFIGS[item]["llm"].metadata_fallback_enabled
        for item in selected
    )
    if needs_llm and generate_json is None:
        print("GEMINI_API_KEY missing or Gemini client failed; LLM configs will fall back to classical search.")

    db = SessionLocal()
    results = {}
    try:
        for name in selected:
            results[name] = run_configuration(name, dataset, db, generate_json)
            print_report(results[name]["summary"], results[name]["failures"])
    finally:
        db.close()

    comparison = [comparison_row(name, results[name]["summary"]) for name in selected]
    if "baseline" in results:
        baseline_violations = results["baseline"]["hard_filter_violations"]
        for name, result in results.items():
            result["filter_compliance_regressed"] = (
                result["hard_filter_violations"] > baseline_violations
            )

    output = {
        "dataset": str(DATASET_PATH),
        "limit": LIMIT,
        "pricing": {
            "input_usd_per_million": LLM_INPUT_USD_PER_MILLION,
            "output_usd_per_million": LLM_OUTPUT_USD_PER_MILLION,
            "model": "gemini-2.5-flash",
        },
        "comparison": comparison,
        "configurations": {
            name: {
                "config": result["config"],
                "summary": result["summary"],
                "hard_filter_violations": result["hard_filter_violations"],
                "filter_compliance_regressed": result.get("filter_compliance_regressed"),
                "failures": result["failures"],
                "cases": result["cases"],
            }
            for name, result in results.items()
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            results.get("name_ranking")
            or results.get("baseline")
            or results[selected[0]],
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    COMPARISON_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print_comparison_table(comparison)
    print()
    print(f"Baseline-style results saved to: {RESULTS_PATH}")
    print(f"Full comparison saved to: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
