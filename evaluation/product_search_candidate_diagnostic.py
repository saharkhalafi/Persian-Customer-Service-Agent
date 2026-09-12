"""Diagnostic-only candidate vs ranking analysis.

Does not change production search. Loads the eval dataset read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app.core.database import SessionLocal
from app.repositories.product_repository import ProductRepository
from app.services.product_llm_config import ProductLlmConfig
from app.services.product_service import ProductService
from evaluation.product_search_eval import (
    _codes,
    _relevance,
    brands_equivalent,
    gender_matches,
    hit_at_k,
    infer_evaluation_type,
    item_matches_filters,
    mrr_score,
    ndcg_at_k,
    parse_price,
)


DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "Eval_product_search.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results" / "product_search_candidate_diagnostic.json"
WATCHED_IDS = {
    "product_eval_023",
    "product_eval_031",
    "product_eval_034",
    "product_eval_035",
    "product_eval_038",
    "product_eval_076",
    "product_eval_077",
    "product_eval_078",
    "product_eval_080",
    "product_eval_082",
    "product_eval_083",
}
CANDIDATE_KS = (20, 50, 100, 500)
RANK_KS = (3, 5, 10, 20, 50, 100)
NARROW_REPORT_TYPES = {
    "DIRECT_PRODUCT",
    "BRAND",
    "CATEGORY",
    "BRAND_CATEGORY",
    "PRICE_FILTER",
    "MULTI_ATTRIBUTE",
    "GENDER_FILTER",
    "AVAILABILITY_FILTER",
    "ATTRIBUTE",
    "COLLOQUIAL",
    "PARTIAL_PRODUCT_NAME",
    "PERSIAN_VARIATION",
}


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def build_production_service(db, name_matching_enabled: bool) -> ProductService:
    return ProductService(
        ProductRepository(db),
        llm_config=ProductLlmConfig(
            query_understanding_enabled=False,
            rerank_enabled=False,
        ),
        name_matching_enabled=name_matching_enabled,
        negative_constraints_enabled=True,
    )


def extracted_filter_exclusions(gold: dict, metadata: dict) -> list[str]:
    reasons: list[str] = []
    if metadata.get("brand") and not brands_equivalent(gold.get("brand"), metadata.get("brand")):
        reasons.append("brand")
    category = metadata.get("category")
    if category:
        sql_fields = [gold.get("type"), gold.get("category_level1"), gold.get("category_level2")]
        if not any(str(category).casefold() in str(field or "").casefold() for field in sql_fields):
            reasons.append("category")
    if metadata.get("color") and str(metadata["color"]).casefold() not in str(
        gold.get("color") or ""
    ).casefold():
        reasons.append("color")
    if metadata.get("gender") and not gender_matches(gold.get("gender"), metadata.get("gender")):
        reasons.append("gender")
    price = parse_price(gold.get("sum_of_price"))
    if metadata.get("price_min") is not None and (
        price is None or price < float(metadata["price_min"])
    ):
        reasons.append("price_min")
    if metadata.get("price_max") is not None and (
        price is None or price > float(metadata["price_max"])
    ):
        reasons.append("price_max")
    if metadata.get("availability") is True and str(gold.get("last_status") or "") != "Enable":
        reasons.append("availability")
    for excluded in metadata.get("exclude_brands") or ():
        if brands_equivalent(gold.get("brand"), excluded):
            reasons.append("exclude_brand")
            break
    return reasons


def classify_narrow_case(
    gold_codes: list[str],
    catalog: dict[str, dict],
    expected_filters: dict,
    metadata: dict,
    gold_in_filtered: list[str],
    best_rank: int | None,
    token_filters: list[str],
    apply_token_filters: bool,
) -> tuple[str | None, list[str]]:
    evidence: list[str] = []
    present = [code for code in gold_codes if code in catalog]
    missing = [code for code in gold_codes if code not in catalog]
    if missing:
        evidence.append(f"gold codes missing from catalog: {missing}")
    if not present:
        return "NO_CATALOG_MATCH", evidence

    matching_expected = [
        code
        for code in present
        if not expected_filters or item_matches_filters(catalog[code], expected_filters)
    ]
    if expected_filters and not matching_expected:
        evidence.append("all catalog golds fail the benchmark expected_filters")
        return "DATA/GOLD_ISSUE", evidence

    in_filtered = set(gold_in_filtered)
    if not in_filtered:
        exclusions = []
        for code in matching_expected or present:
            reasons = extracted_filter_exclusions(catalog[code], metadata)
            if reasons:
                exclusions.append(f"{code}:{','.join(reasons)}")
        if exclusions:
            evidence.append(f"extracted hard filters exclude gold: {exclusions}")
            return "FILTER_FAILURE", evidence
        if apply_token_filters and token_filters:
            evidence.append(f"leftover token filters applied: {token_filters}")
        evidence.append("gold is in catalog and passes extracted metadata but not the filtered SQL set")
        return "CANDIDATE_GENERATION_FAILURE", evidence

    if best_rank is None or best_rank > 10:
        evidence.append(
            f"gold in hard-filtered set {sorted(in_filtered)}; best rank={best_rank}"
        )
        return "RANKING_FAILURE", evidence
    return None, evidence


def _best_rank(gold_codes: list[str], ranked_codes: list[str]) -> int | None:
    ranks = []
    index = {code: rank for rank, code in enumerate(ranked_codes, start=1)}
    for code in gold_codes:
        if code in index:
            ranks.append(index[code])
    return min(ranks) if ranks else None


def _candidate_recall(gold_codes: list[str], ranked_codes: list[str], k: int) -> bool:
    return any(code in set(ranked_codes[:k]) for code in gold_codes)


def diagnose_item(service: ProductService, item: dict) -> dict:
    query = item["query"]
    gold_codes = _codes(item)
    judged = _relevance(item)
    relevant = [code for code, grade in judged.items() if grade > 0] or gold_codes
    eval_type = infer_evaluation_type(item)
    diagnostic = service.search_with_diagnostics(query, gold_codes=relevant)
    catalog = service.repository.fetch_products_by_codes(relevant)
    best_rank = _best_rank(relevant, diagnostic["ranked_codes"])
    gold_in_filtered = diagnostic["gold_in_filtered"]
    classification = None
    evidence: list[str] = []
    if eval_type == "narrow" and relevant:
        classification, evidence = classify_narrow_case(
            gold_codes=relevant,
            catalog=catalog,
            expected_filters=item.get("expected_filters") or {},
            metadata=diagnostic["metadata"],
            gold_in_filtered=gold_in_filtered,
            best_rank=best_rank,
            token_filters=diagnostic["token_filters"],
            apply_token_filters=diagnostic["apply_token_filters"],
        )
    gold_records = []
    for code in relevant:
        product = catalog.get(code)
        gold_records.append(
            {
                "product_code": code,
                "in_catalog": product is not None,
                "in_filtered": code in gold_in_filtered,
                "rank": _best_rank([code], diagnostic["ranked_codes"]),
                "product_name": None if product is None else product.get("product_name"),
                "brand": None if product is None else product.get("brand"),
                "type": None if product is None else product.get("type"),
                "extracted_exclusions": (
                    []
                    if product is None
                    else extracted_filter_exclusions(product, diagnostic["metadata"])
                ),
            }
        )
    ranking_flags = {
        f"rank<={k}": (best_rank is not None and best_rank <= k) for k in RANK_KS
    }
    candidate_flags = {
        f"candidate_recall@{k}": _candidate_recall(relevant, diagnostic["ranked_codes"], k)
        for k in CANDIDATE_KS
    }
    return {
        "id": item.get("id"),
        "query": query,
        "query_type": item.get("query_type"),
        "evaluation_type": eval_type,
        "expected_filters": item.get("expected_filters") or {},
        "gold_codes": relevant,
        "normalized_query": diagnostic["normalized_query"],
        "metadata": diagnostic["metadata"],
        "search_tokens": diagnostic["search_tokens"],
        "hard_sql_filters": diagnostic["hard_sql_filters"],
        "apply_token_filters": diagnostic["apply_token_filters"],
        "token_filters": diagnostic["token_filters"],
        "hard_filtered_count": diagnostic["hard_filtered_count"],
        "restrict_name_pool": diagnostic["restrict_name_pool"],
        "gold_in_filtered_candidates": bool(gold_in_filtered),
        "gold_in_filtered": gold_in_filtered,
        "gold_rank": best_rank,
        "gold_products": gold_records,
        "classification": classification,
        "evidence": evidence,
        "name_matching_enabled": diagnostic["name_matching_enabled"],
        "top_3": diagnostic["top_3"],
        "top_5": diagnostic["top_5"],
        "top_10": diagnostic["top_10"],
        "outranked_by": diagnostic["ranked_preview"][:5],
        "hit@3": hit_at_k(diagnostic["ranked_codes"], set(relevant), 3) if relevant else None,
        "hit@5": hit_at_k(diagnostic["ranked_codes"], set(relevant), 5) if relevant else None,
        "hit@10": hit_at_k(diagnostic["ranked_codes"], set(relevant), 10) if relevant else None,
        "mrr": mrr_score(diagnostic["ranked_codes"], set(relevant)) if relevant else None,
        "ndcg@10": ndcg_at_k(diagnostic["ranked_codes"], judged, 10) if judged else None,
        **ranking_flags,
        **candidate_flags,
        "watched": item.get("id") in WATCHED_IDS,
    }


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate_narrow(cases: list[dict]) -> dict:
    narrow = [
        case
        for case in cases
        if case["evaluation_type"] == "narrow" and case.get("gold_codes")
    ]
    classes: dict[str, int] = defaultdict(int)
    for case in narrow:
        if case.get("classification"):
            classes[case["classification"]] += 1
        elif (case.get("hit@10") or 0) < 1:
            classes["UNCLASSIFIED_FAIL"] += 1
    failed = [case for case in narrow if (case.get("hit@10") or 0) < 1]
    failed_with_gold_in_filter = [
        case for case in failed if case.get("gold_in_filtered_candidates")
    ]
    return {
        "queries_evaluated": len(narrow),
        "gold_in_filtered_set": mean(
            [1.0 if case.get("gold_in_filtered_candidates") else 0.0 for case in narrow]
        ),
        "candidate_recall@20": mean(
            [1.0 if case.get("candidate_recall@20") else 0.0 for case in narrow]
        ),
        "candidate_recall@50": mean(
            [1.0 if case.get("candidate_recall@50") else 0.0 for case in narrow]
        ),
        "candidate_recall@100": mean(
            [1.0 if case.get("candidate_recall@100") else 0.0 for case in narrow]
        ),
        "candidate_recall@500": mean(
            [1.0 if case.get("candidate_recall@500") else 0.0 for case in narrow]
        ),
        "Hit@3": mean([case.get("hit@3") for case in narrow]),
        "Hit@5": mean([case.get("hit@5") for case in narrow]),
        "Hit@10": mean([case.get("hit@10") for case in narrow]),
        "MRR": mean([case.get("mrr") for case in narrow]),
        "NDCG@10": mean([case.get("ndcg@10") for case in narrow]),
        "failed_narrow": len(failed),
        "failed_with_gold_in_filtered": len(failed_with_gold_in_filter),
        "failed_gold_in_filtered_pct": (
            len(failed_with_gold_in_filter) / len(failed) if failed else None
        ),
        "classifications": dict(sorted(classes.items())),
    }


def aggregate_by_category(cases: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        if case["evaluation_type"] != "narrow" or not case.get("gold_codes"):
            continue
        grouped[case.get("query_type") or "unknown"].append(case)
    report = {}
    for name in sorted(NARROW_REPORT_TYPES | set(grouped)):
        group = grouped.get(name) or []
        if not group:
            continue
        classes: dict[str, int] = defaultdict(int)
        for case in group:
            if case.get("classification"):
                classes[case["classification"]] += 1
        main = None
        if classes:
            main = max(classes.items(), key=lambda item: item[1])[0]
        report[name] = {
            "count": len(group),
            "gold_in_filtered_set": mean(
                [1.0 if case.get("gold_in_filtered_candidates") else 0.0 for case in group]
            ),
            "candidate_recall@20": mean(
                [1.0 if case.get("candidate_recall@20") else 0.0 for case in group]
            ),
            "candidate_recall@50": mean(
                [1.0 if case.get("candidate_recall@50") else 0.0 for case in group]
            ),
            "candidate_recall@100": mean(
                [1.0 if case.get("candidate_recall@100") else 0.0 for case in group]
            ),
            "candidate_recall@500": mean(
                [1.0 if case.get("candidate_recall@500") else 0.0 for case in group]
            ),
            "Hit@10": mean([case.get("hit@10") for case in group]),
            "MRR": mean([case.get("mrr") for case in group]),
            "classifications": dict(sorted(classes.items())),
            "main_failure_type": main,
        }
    return report


def compare_name_matching(production: list[dict], named: list[dict]) -> dict:
    by_id = {case["id"]: case for case in named}
    removed = []
    pushed_down = []
    improved = []
    for case in production:
        other = by_id.get(case["id"])
        if not other or case["evaluation_type"] != "narrow":
            continue
        if case.get("gold_in_filtered_candidates") and not other.get(
            "gold_in_filtered_candidates"
        ):
            removed.append(case["id"])
        prod_rank = case.get("gold_rank")
        name_rank = other.get("gold_rank")
        if prod_rank and name_rank and name_rank > prod_rank:
            pushed_down.append(
                {"id": case["id"], "production_rank": prod_rank, "name_rank": name_rank}
            )
        if (other.get("hit@10") or 0) > (case.get("hit@10") or 0):
            improved.append(case["id"])
        elif prod_rank and name_rank and name_rank < prod_rank and name_rank <= 10:
            if case["id"] not in improved:
                improved.append(case["id"])
    return {
        "gold_removed_from_filtered_or_effective_set": removed,
        "gold_pushed_down": pushed_down,
        "hit_or_rank_improved": improved,
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} ({value * 100:.2f}%)"


def print_report(payload: dict) -> None:
    aggregate = payload["aggregate"]
    print("=== Product Search Candidate Diagnostic ===")
    print()
    print("Metric | Result")
    rows = [
        ("Queries evaluated", aggregate["queries_evaluated"]),
        ("Gold-in-filtered-set", fmt(aggregate["gold_in_filtered_set"])),
        ("Candidate Recall@20", fmt(aggregate["candidate_recall@20"])),
        ("Candidate Recall@50", fmt(aggregate["candidate_recall@50"])),
        ("Candidate Recall@100", fmt(aggregate["candidate_recall@100"])),
        ("Candidate Recall@500", fmt(aggregate["candidate_recall@500"])),
        ("Hit@10", fmt(aggregate["Hit@10"])),
        ("Hit@5", fmt(aggregate["Hit@5"])),
        ("Hit@3", fmt(aggregate["Hit@3"])),
        ("MRR", fmt(aggregate["MRR"])),
        ("NDCG@10", fmt(aggregate["NDCG@10"])),
        (
            "Ranking failures",
            aggregate["classifications"].get("RANKING_FAILURE", 0),
        ),
        (
            "Candidate-generation failures",
            aggregate["classifications"].get("CANDIDATE_GENERATION_FAILURE", 0),
        ),
        (
            "Filter failures",
            aggregate["classifications"].get("FILTER_FAILURE", 0),
        ),
        (
            "Gold/data issues",
            aggregate["classifications"].get("DATA/GOLD_ISSUE", 0),
        ),
        (
            "No catalog match",
            aggregate["classifications"].get("NO_CATALOG_MATCH", 0),
        ),
        (
            "Failed with gold in filtered set",
            fmt(aggregate["failed_gold_in_filtered_pct"]),
        ),
    ]
    for name, value in rows:
        print(f"{name} | {value}")
    print()
    print("Category | Candidate Coverage | Hit@10 | Main Failure Type")
    for name, stats in payload["by_category"].items():
        print(
            f"{name} n={stats['count']} | "
            f"{fmt(stats['gold_in_filtered_set'])} | "
            f"{fmt(stats['Hit@10'])} | "
            f"{stats.get('main_failure_type') or 'n/a'}"
        )
    print()
    print("Watched queries:")
    for case in payload["watched"]:
        print(
            f"- {case['id']} [{case.get('classification')}] {case['query']} "
            f"filtered={case['hard_filtered_count']} "
            f"gold_in={case['gold_in_filtered_candidates']} "
            f"rank={case.get('gold_rank')}"
        )


def run_pipeline(name: str, dataset: list[dict], name_matching: bool) -> dict:
    db = SessionLocal()
    try:
        service = build_production_service(db, name_matching_enabled=name_matching)
        cases = []
        for index, item in enumerate(dataset, start=1):
            print(f"[{name} {index}/{len(dataset)}] {item.get('id')}")
            cases.append(diagnose_item(service, item))
    finally:
        db.close()
    return {
        "name": name,
        "name_matching_enabled": name_matching,
        "negative_constraints_enabled": True,
        "aggregate": aggregate_narrow(cases),
        "by_category": aggregate_by_category(cases),
        "watched": [case for case in cases if case.get("watched")],
        "queries": cases,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare-name-matching",
        action="store_true",
        help="Also run optional product-name matching as a separate diagnostic.",
    )
    args = parser.parse_args()
    dataset = load_dataset()
    production = run_pipeline("production", dataset, name_matching=False)
    output = {
        "dataset": str(DATASET_PATH),
        "production": production,
    }
    print_report(production)
    if args.compare_name_matching:
        named = run_pipeline("name_ranking", dataset, name_matching=True)
        output["name_ranking"] = named
        output["name_ranking_comparison"] = compare_name_matching(
            production["queries"],
            named["queries"],
        )
        print()
        print("=== Optional name-matching diagnostic ===")
        print_report(named)
        print()
        print("Name-matching comparison:")
        print(json.dumps(output["name_ranking_comparison"], ensure_ascii=False, indent=2))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
