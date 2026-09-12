"""Isolated Dense vs Hybrid product-search experiment.

Does not change production ProductService, ranking, metadata, or flags.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.core.database import SessionLocal
from app.repositories.product_repository import ProductRepository
from app.services.product_brand_aliases import brand_match_keys
from app.services.product_llm_config import ProductLlmConfig
from app.services.product_llm_ranker import ProductLlmRanker, gemini_generate_json
from app.services.product_metadata import (
    ProductSearchMetadata,
    search_tokens_for_metadata,
)
from app.services.product_query_normalizer import fold_search_text, hard_catalog_tokens
from app.services.product_service import ProductService
from evaluation.product_search_eval import (
    KS,
    LIMIT,
    annotate_dataset,
    attach_db_facts,
    aggregate,
    classify_failure,
    filter_compliance_at_k,
    family_hit_at_k,
    hard_filter_violations,
    hit_at_k,
    infer_evaluation_type,
    metadata_field_accuracy,
    mrr_score,
    ndcg_at_k,
    parse_price,
    precision_at_k,
    recall_at_k,
    try_gemini_generator,
    _relevance,
    _codes,
    BROAD_GOLD_THRESHOLD,
    print_report,
    print_comparison_table,
    comparison_row,
)

RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "product_search_dense_hybrid.json"
EMBED_CACHE = PROJECT_ROOT / "evaluation" / "results" / "product_dense_embeddings.npz"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
CANDIDATE_K = 50
DENSE_WEIGHT = 0.6
LEXICAL_WEIGHT = 0.4
BM25_K1 = 1.5
BM25_B = 0.75
FOCUS_TYPES = (
    "BRAND_CATEGORY",
    "PERSIAN_VARIATION",
    "COLLOQUIAL",
    "PARTIAL_PRODUCT_NAME",
)


def production_llm_config() -> ProductLlmConfig:
    return ProductLlmConfig(
        query_understanding_enabled=False,
        rerank_enabled=False,
        metadata_fallback_enabled=True,
        candidate_k=20,
        final_k=10,
        confidence_threshold=0.7,
    )


def metadata_from_dict(raw: dict | None) -> ProductSearchMetadata:
    data = raw or {}
    return ProductSearchMetadata(
        brand=data.get("brand"),
        category=data.get("category"),
        color=data.get("color"),
        material=data.get("material"),
        gender=data.get("gender"),
        price_min=data.get("price_min"),
        price_max=data.get("price_max"),
        availability=data.get("availability"),
        exclude_brands=tuple(data.get("exclude_brands") or ()),
    )


def searchable_text(row: dict) -> str:
    return " ".join(
        str(row.get(field) or "")
        for field in (
            "product_name",
            "brand",
            "type",
            "sku",
            "product_code",
            "category_level1",
            "category_level2",
            "color",
        )
    )


def tokenize(text: str) -> list[str]:
    folded = fold_search_text(text).casefold()
    return [token for token in folded.split() if token]


def load_catalog(db) -> list[dict]:
    rows = db.execute(
        text(
            """
            SELECT
                product_code, product_name, brand, type, sku,
                category_level1, category_level2, color, gender,
                sum_of_price, special_price, last_status
            FROM public.products
            WHERE product_code IS NOT NULL
            """
        )
    ).mappings()
    catalog = []
    for row in rows:
        item = dict(row)
        item["product_code"] = str(item.get("product_code") or "")
        item["_search"] = searchable_text(item)
        item["_search_fold"] = fold_search_text(item["_search"]).casefold()
        item["_tokens"] = tokenize(item["_search"])
        catalog.append(item)
    return catalog


def row_matches_metadata(
    row: dict,
    metadata: ProductSearchMetadata,
    tokens: list[str],
) -> bool:
    if metadata.brand:
        brand = fold_search_text(str(row.get("brand") or "")).casefold()
        if brand not in set(brand_match_keys(metadata.brand)):
            return False
    if metadata.exclude_brands:
        brand = fold_search_text(str(row.get("brand") or "")).casefold()
        excluded = {
            key
            for name in metadata.exclude_brands
            for key in brand_match_keys(name)
        }
        if brand and brand in excluded:
            return False
    if metadata.category:
        needle = metadata.category.casefold()
        fields = [
            str(row.get("type") or ""),
            str(row.get("category_level1") or ""),
            str(row.get("category_level2") or ""),
        ]
        if not any(needle in field.casefold() for field in fields):
            return False
    if metadata.color and metadata.color.casefold() not in str(row.get("color") or "").casefold():
        return False
    if metadata.material and metadata.material.casefold() not in row["_search_fold"]:
        return False
    if metadata.gender and metadata.gender.casefold() not in str(row.get("gender") or "").casefold():
        return False
    price = parse_price(row.get("sum_of_price"))
    if metadata.price_min is not None and (price is None or price < metadata.price_min):
        return False
    if metadata.price_max is not None and (price is None or price > metadata.price_max):
        return False
    if metadata.availability is True and str(row.get("last_status") or "") != "Enable":
        return False
    if metadata.availability is False and str(row.get("last_status") or "") == "Enable":
        return False
    apply_token_filters = bool(tokens) and not metadata.has_positive_filters()
    if apply_token_filters:
        haystack = row["_search_fold"]
        for token in hard_catalog_tokens(tokens)[:12]:
            if fold_search_text(token).casefold() not in haystack:
                return False
    return True


def filtered_indices(
    catalog: list[dict],
    metadata: ProductSearchMetadata,
    tokens: list[str],
) -> list[int]:
    return [
        index
        for index, row in enumerate(catalog)
        if row_matches_metadata(row, metadata, tokens)
    ]


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if math.isclose(high, low):
        return np.ones_like(values)
    return (values - low) / (high - low)


def bm25_scores(
    query_tokens: list[str],
    doc_tokens: list[list[str]],
    indices: list[int],
    idf: dict[str, float],
    avgdl: float,
) -> np.ndarray:
    scores = np.zeros(len(indices), dtype=np.float32)
    query = [token.casefold() for token in query_tokens if token]
    if not query or not indices:
        return scores
    for offset, index in enumerate(indices):
        tokens = doc_tokens[index]
        length = len(tokens) or 1
        counts = Counter(tokens)
        score = 0.0
        for token in query:
            freq = counts.get(token, 0)
            if not freq:
                continue
            denom = freq + BM25_K1 * (1.0 - BM25_B + BM25_B * length / avgdl)
            score += idf.get(token, 0.0) * (freq * (BM25_K1 + 1.0)) / denom
        scores[offset] = score
    return scores


def build_bm25_stats(doc_tokens: list[list[str]]) -> tuple[dict[str, float], float]:
    document_count = max(len(doc_tokens), 1)
    df: Counter[str] = Counter()
    total_len = 0
    for tokens in doc_tokens:
        total_len += len(tokens)
        df.update(set(tokens))
    avgdl = total_len / document_count if document_count else 1.0
    idf = {
        token: math.log(1.0 + (document_count - count + 0.5) / (count + 0.5))
        for token, count in df.items()
    }
    return idf, max(avgdl, 1.0)


def classical_scores(
    query: str,
    catalog: list[dict],
    indices: list[int],
    metadata: ProductSearchMetadata,
    tokens: list[str],
) -> np.ndarray:
    folded_query = fold_search_text(query).casefold()
    query_tokens = [fold_search_text(token).casefold() for token in tokens if token]
    scores = np.zeros(len(indices), dtype=np.float32)
    for offset, index in enumerate(indices):
        row = catalog[index]
        name = fold_search_text(str(row.get("product_name") or "")).casefold()
        score = 0.0
        if name and name == folded_query:
            score += 100.0
        if metadata.brand and fold_search_text(str(row.get("brand") or "")).casefold() in set(
            brand_match_keys(metadata.brand)
        ):
            score += 50.0
        if metadata.category and metadata.category.casefold() in str(row.get("type") or "").casefold():
            score += 40.0
        if folded_query and folded_query in name:
            score += 25.0
        if query_tokens:
            hits = sum(1 for token in query_tokens if token in row["_search_fold"])
            score += hits * 8.0
            if hits == len(query_tokens):
                score += 20.0
        scores[offset] = score
    return scores


def topk_indices(scores: np.ndarray, indices: list[int], k: int) -> list[int]:
    if not indices:
        return []
    order = np.argsort(-scores, kind="stable")[:k]
    return [indices[int(position)] for position in order]


def score_experiment_case(
    item: dict,
    retrieved_items: list[dict],
    extracted: dict,
    latency_ms: float,
    extra: dict | None = None,
) -> dict:
    eval_type = infer_evaluation_type(item)
    filters = item.get("expected_filters") or {}
    judged = _relevance(item)
    relevant = {code for code, grade in judged.items() if grade > 0}
    exhaustive = eval_type == "narrow" and bool(relevant) and len(relevant) < BROAD_GOLD_THRESHOLD
    retrieved_codes = [
        str(row.get("product_code") or "")
        for row in retrieved_items
        if row.get("product_code")
    ]
    gold_items = [
        row for row in retrieved_items if False
    ]
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
        ranking["family_hit@3"] = family_hit_at_k(retrieved_items, extra.get("gold_items") if extra else [], 3)
        ranking["family_hit@5"] = family_hit_at_k(retrieved_items, extra.get("gold_items") if extra else [], 5)
        ranking["family_hit@10"] = family_hit_at_k(retrieved_items, extra.get("gold_items") if extra else [], 10)

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
        "query": item.get("query"),
        "query_type": item.get("query_type"),
        "evaluation_type": eval_type,
        "expected_filters": filters,
        "expected_codes": _codes(item),
        "extracted": extracted,
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
        "latency_ms": latency_ms,
        "llm_calls": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_query_understanding_used": False,
        "llm_rerank_used": False,
    }
    if extra:
        case.update(extra)
    case["failure_reason"] = classify_failure(item, case)
    return case


def e5_texts(kind: str, texts: list[str]) -> list[str]:
    prefix = "query: " if kind == "query" else "passage: "
    return [prefix + text for text in texts]


def load_or_build_embeddings(model, catalog: list[dict], model_name: str) -> tuple[np.ndarray, float]:
    texts = [row["_search"] for row in catalog]
    codes = [row["product_code"] for row in catalog]
    if EMBED_CACHE.exists():
        payload = np.load(EMBED_CACHE, allow_pickle=True)
        cached_codes = list(payload["codes"])
        if (
            str(payload["model_name"]) == model_name
            and cached_codes == codes
        ):
            print(f"Loaded cached embeddings {payload['vectors'].shape} from {EMBED_CACHE}")
            return payload["vectors"], 0.0
    started = time.perf_counter()
    vectors = model.encode(
        e5_texts("passage", texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    elapsed = time.perf_counter() - started
    array = np.asarray(vectors, dtype=np.float32)
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBED_CACHE,
        vectors=array,
        codes=np.asarray(codes),
        model_name=np.asarray(model_name),
    )
    return array, elapsed


def retrieve(
    mode: str,
    query: str,
    query_vector: np.ndarray,
    catalog: list[dict],
    vectors: np.ndarray,
    metadata: ProductSearchMetadata,
    tokens: list[str],
    doc_tokens: list[list[str]],
    idf: dict[str, float],
    avgdl: float,
) -> tuple[list[dict], dict]:
    started = time.perf_counter()
    pool = filtered_indices(catalog, metadata, tokens)
    gold_in_filtered = None
    if not pool:
        return [], {
            "filtered_count": 0,
            "candidate_recall@50": 0.0,
            "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
        }
    dense = vectors[pool] @ query_vector
    lexical = bm25_scores(tokens or tokenize(query), doc_tokens, pool, idf, avgdl)
    if mode == "dense":
        ranked = topk_indices(dense, pool, LIMIT)
        candidate_pool = topk_indices(dense, pool, CANDIDATE_K)
    else:
        fused = (
            DENSE_WEIGHT * min_max_normalize(dense)
            + LEXICAL_WEIGHT * min_max_normalize(lexical)
        )
        hybrid_pool = topk_indices(fused, pool, CANDIDATE_K)
        classical = classical_scores(query, catalog, hybrid_pool, metadata, tokens)
        ranked = topk_indices(classical, hybrid_pool, LIMIT)
        candidate_pool = hybrid_pool
    items = [catalog[index] for index in ranked]
    return items, {
        "filtered_count": len(pool),
        "candidate_pool": [catalog[index]["product_code"] for index in candidate_pool],
        "retrieval_latency_ms": (time.perf_counter() - started) * 1000,
    }


def compare_focus(prod_cases: list[dict], other_cases: list[dict]) -> dict:
    prod = {case["id"]: case for case in prod_cases}
    other = {case["id"]: case for case in other_cases}
    improved = []
    regressed = []
    unchanged = []
    for case_id, baseline in prod.items():
        if baseline.get("query_type") not in FOCUS_TYPES and baseline.get("failure_reason") != "METADATA_MISS":
            if other.get(case_id, {}).get("query_type") not in FOCUS_TYPES:
                continue
        current = other[case_id]
        before = baseline.get("hit@10")
        after = current.get("hit@10")
        filt_before = baseline.get("filter_compliance@10")
        filt_after = current.get("filter_compliance@10")
        payload = {
            "id": case_id,
            "query": baseline.get("query"),
            "query_type": baseline.get("query_type"),
            "hit@10": [before, after],
            "mrr": [baseline.get("mrr"), current.get("mrr")],
            "filter_compliance@10": [filt_before, filt_after],
            "failure": [baseline.get("failure_reason"), current.get("failure_reason")],
        }
        better = (after or 0) > (before or 0) or (
            (after or 0) == (before or 0)
            and (current.get("mrr") or 0) > (baseline.get("mrr") or 0)
        )
        worse = (after or 0) < (before or 0) or (
            (filt_after is not None and filt_before is not None and filt_after + 1e-9 < filt_before)
            and (after or 0) <= (before or 0)
        )
        if better and not worse:
            improved.append(payload)
        elif worse:
            regressed.append(payload)
        else:
            unchanged.append(payload)
    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
    }


def candidate_recall(cases: list[dict]) -> float | None:
    values = [
        case.get("candidate_recall@50")
        for case in cases
        if case.get("evaluation_type") == "narrow" and case.get("candidate_recall@50") is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    dataset = annotate_dataset()
    db = SessionLocal()
    generate_json = try_gemini_generator()
    service = ProductService(
        ProductRepository(db),
        llm_ranker=ProductLlmRanker(generate_json=generate_json, config=production_llm_config()),
        llm_config=production_llm_config(),
        name_matching_enabled=False,
        negative_constraints_enabled=True,
    )

    from evaluation.product_search_eval import evaluate_case

    production_cases = []
    print("=== Production baseline ===")
    for index, item in enumerate(dataset, start=1):
        print(f"[production {index}/{len(dataset)}] {item['id']}")
        production_cases.append(evaluate_case(service, db, item))
    production_summary = aggregate(production_cases)

    print("Loading catalog...")
    catalog = load_catalog(db)
    gold_by_code = {row["product_code"]: row for row in catalog}
    doc_tokens = [row["_tokens"] for row in catalog]
    idf, avgdl = build_bm25_stats(doc_tokens)

    from sentence_transformers import SentenceTransformer

    print(f"Loading {args.model}...")
    model_started = time.perf_counter()
    model = SentenceTransformer(args.model)
    model_load_s = time.perf_counter() - model_started
    vectors, build_s = load_or_build_embeddings(model, catalog, args.model)
    embedding_bytes = int(vectors.nbytes)
    try:
        model_bytes = sum(
            int(param.nbytes) for param in model[0].auto_model.parameters()
        )
    except Exception:
        model_bytes = 0

    queries = [item["query"] for item in dataset]
    query_started = time.perf_counter()
    query_vectors = model.encode(
        e5_texts("query", queries),
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_encode_ms = (time.perf_counter() - query_started) * 1000 / max(len(queries), 1)

    results = {"production": production_cases}
    for mode in ("dense", "hybrid"):
        print(f"=== {mode} ===")
        cases = []
        for index, (item, baseline) in enumerate(zip(dataset, production_cases), start=1):
            print(f"[{mode} {index}/{len(dataset)}] {item['id']}")
            metadata = metadata_from_dict(baseline.get("extracted") or {})
            tokens = search_tokens_for_metadata(item["query"], metadata)
            retrieved, info = retrieve(
                mode,
                item["query"],
                np.asarray(query_vectors[index - 1], dtype=np.float32),
                catalog,
                vectors,
                metadata,
                tokens,
                doc_tokens,
                idf,
                avgdl,
            )
            relevant = {code for code, grade in _relevance(item).items() if grade > 0}
            gold_items = [gold_by_code[code] for code in relevant if code in gold_by_code]
            filtered_codes = set()
            if info.get("filtered_count"):
                pool = filtered_indices(catalog, metadata, tokens)
                filtered_codes = {catalog[i]["product_code"] for i in pool}
            extra = {
                "gold_items": gold_items,
                "filtered_count": info.get("filtered_count"),
                "candidate_recall@50": (
                    1.0
                    if relevant and any(code in set(info.get("candidate_pool") or []) for code in relevant)
                    else (0.0 if relevant else None)
                ),
                "gold_in_filtered": (
                    1.0
                    if relevant and any(code in filtered_codes for code in relevant)
                    else (0.0 if relevant else None)
                ),
                "retrieval_latency_ms": info.get("retrieval_latency_ms"),
                "llm_metadata_fallback_used": baseline.get("llm_metadata_fallback_used"),
            }
            latency = float(info.get("retrieval_latency_ms") or 0) + query_encode_ms
            cases.append(
                score_experiment_case(
                    item,
                    retrieved,
                    baseline.get("extracted") or {},
                    latency,
                    extra,
                )
            )
        results[mode] = cases

    db.close()

    summaries = {
        name: aggregate(cases)
        for name, cases in results.items()
    }
    for name, cases in results.items():
        summaries[name]["hard_filter_violations"] = hard_filter_violations(cases)
        summaries[name]["candidate_recall@50"] = candidate_recall(cases)
        summaries[name]["gold_in_filtered"] = candidate_recall(
            [
                {**case, "candidate_recall@50": case.get("gold_in_filtered")}
                for case in cases
            ]
        )

    comparison = [comparison_row(name, summaries[name]) for name in ("production", "dense", "hybrid")]
    analysis = {
        "dense": compare_focus(results["production"], results["dense"]),
        "hybrid": compare_focus(results["production"], results["hybrid"]),
    }
    resources = {
        "model": args.model,
        "catalog_size": len(catalog),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        "embedding_build_seconds": build_s,
        "model_load_seconds": model_load_s,
        "approx_embedding_memory_mb": embedding_bytes / (1024 * 1024),
        "approx_model_memory_mb": model_bytes / (1024 * 1024) if model_bytes else None,
        "query_encode_ms_average": query_encode_ms,
        "candidate_k": CANDIDATE_K,
        "fusion": {
            "dense_weight": DENSE_WEIGHT,
            "lexical_weight": LEXICAL_WEIGHT,
            "strategy": "minmax(dense)*0.6 + minmax(BM25)*0.4, then classical rerank of top-50",
        },
        "additional_dependencies": [
            "sentence-transformers (already present)",
            "numpy (already present)",
            "no FAISS / new database / production config change",
        ],
    }

    output = {
        "dataset": str(PROJECT_ROOT / "Evaluate_data" / "Eval_product_search.json"),
        "resources": resources,
        "comparison": comparison,
        "summaries": summaries,
        "focus": {
            name: summaries[name].get("focus")
            for name in ("production", "dense", "hybrid")
        },
        "analysis": analysis,
        "cases": {
            name: cases
            for name, cases in results.items()
        },
    }
    RESULTS_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name in ("production", "dense", "hybrid"):
        print()
        print(f"===== {name} =====")
        print_report(summaries[name], [case for case in results[name] if not case.get("pass")])
        print(f"  Candidate recall@50: {summaries[name].get('candidate_recall@50')}")
        print(f"  Hard-filter violations: {summaries[name].get('hard_filter_violations')}")
    print_comparison_table(comparison)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
