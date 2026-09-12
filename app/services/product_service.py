from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from app.core.observability import log_event, safe_query_ref
from app.repositories.product_repository import ProductRepository
from app.schemas.product_schemas import ProductSearchItem
from app.services.product_llm_config import ProductLlmConfig, load_product_llm_config
from app.services.product_llm_ranker import (
    LlmUsage,
    ProductLlmRanker,
    needs_query_understanding,
    needs_rerank,
)
from app.services.product_metadata import (
    ProductSearchMetadata,
    extract_product_metadata,
    product_name_phrase,
    product_name_tokens,
    ranking_query_text,
    ranking_tokens_for_query,
    search_tokens_for_metadata,
)
from app.services.product_metadata_confidence import assess_metadata_confidence
from app.services.product_name_matching import (
    load_name_candidate_limit,
    load_name_matching_enabled,
    load_negative_constraints_enabled,
)
from app.services.product_query_normalizer import normalize_product_query


@dataclass
class ProductSearchDebug:
    query: str
    normalized_query: str
    deterministic_metadata: dict
    metadata: dict
    llm_query_understanding_used: bool = False
    llm_metadata_fallback_used: bool = False
    llm_rerank_used: bool = False
    metadata_confidence: float = 1.0
    metadata_confidence_reasons: list[str] = field(default_factory=list)
    db_candidate_count: int = 0
    llm_candidate_count: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    latency_ms: float = 0.0
    fallback_reason: str | None = None
    llm_understanding: dict | None = None
    attributes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProductSearchOutcome:
    items: list[ProductSearchItem]
    debug: ProductSearchDebug


class ProductService:
    def __init__(
        self,
        repository: ProductRepository,
        llm_ranker: ProductLlmRanker | None = None,
        llm_config: ProductLlmConfig | None = None,
        name_matching_enabled: bool | None = None,
        negative_constraints_enabled: bool | None = None,
        name_candidate_limit: int | None = None,
    ):
        self.repository = repository
        self.llm_config = llm_config or load_product_llm_config()
        self.llm_ranker = llm_ranker
        self.name_matching_enabled = (
            load_name_matching_enabled()
            if name_matching_enabled is None
            else name_matching_enabled
        )
        self.negative_constraints_enabled = (
            load_negative_constraints_enabled()
            if negative_constraints_enabled is None
            else negative_constraints_enabled
        )
        self.name_candidate_limit = (
            load_name_candidate_limit()
            if name_candidate_limit is None
            else name_candidate_limit
        )

    def search_products(
        self,
        query: str,
        limit: int = 10,
    ) -> list[ProductSearchItem]:
        return self.search_with_debug(query=query, limit=limit).items

    def search_with_debug(
        self,
        query: str,
        limit: int = 10,
    ) -> ProductSearchOutcome:
        started = time.perf_counter()
        query = query.strip()[:200]
        usage = LlmUsage()
        deterministic = extract_product_metadata(
            query,
            allow_negatives=self.negative_constraints_enabled,
        )
        metadata = deterministic
        attributes: list[str] = []
        llm_understanding_used = False
        llm_metadata_fallback_used = False
        llm_rerank_used = False
        understanding_raw = None
        confidence = assess_metadata_confidence(
            query,
            deterministic,
            search_tokens_for_metadata(query, deterministic),
            threshold=self.llm_config.metadata_confidence_threshold,
        )

        if not query:
            debug = self._debug(
                query=query,
                metadata=metadata,
                deterministic=deterministic,
                started=started,
                usage=usage,
                result_count=0,
            )
            return ProductSearchOutcome(items=[], debug=debug)

        public_limit = min(max(limit, 1), 20)
        tokens = search_tokens_for_metadata(query, metadata)

        use_metadata_fallback = bool(
            self.llm_config.metadata_fallback_enabled and confidence.needs_llm
        )
        use_query_understanding = bool(
            self.llm_config.query_understanding_enabled
            and self.llm_ranker
            and needs_query_understanding(query, metadata, tokens)
        )
        if use_metadata_fallback:
            if self.llm_ranker is None:
                usage.fallback_reason = "llm_unavailable"
            else:
                try:
                    understood = self.llm_ranker.understand_metadata(
                        query, metadata, usage
                    )
                    metadata = understood.metadata
                    understanding_raw = understood.raw or None
                    llm_metadata_fallback_used = usage.calls > 0
                    tokens = search_tokens_for_metadata(query, metadata)
                except Exception:
                    usage.fallback_reason = usage.fallback_reason or "llm_metadata_error"
                    metadata = deterministic
                    tokens = search_tokens_for_metadata(query, metadata)
        elif use_query_understanding:
            understood = self.llm_ranker.understand_query(query, metadata, usage)
            metadata = understood.metadata
            attributes = understood.attributes
            understanding_raw = understood.raw or None
            llm_understanding_used = usage.calls > 0
            tokens = search_tokens_for_metadata(query, metadata)

        if not tokens and not metadata.has_filters():
            debug = self._debug(
                query=query,
                metadata=metadata,
                deterministic=deterministic,
                started=started,
                usage=usage,
                llm_understanding_used=llm_understanding_used,
                llm_metadata_fallback_used=llm_metadata_fallback_used,
                metadata_confidence=confidence.score,
                metadata_confidence_reasons=list(confidence.reasons),
                attributes=attributes,
                understanding_raw=understanding_raw,
                fallback_triggered=use_metadata_fallback,
                result_count=0,
            )
            return ProductSearchOutcome(items=[], debug=debug)

        ranking_tokens = ranking_tokens_for_query(query)
        name_tokens = product_name_tokens(query, metadata)
        name_phrase = product_name_phrase(query, metadata)
        for attribute in attributes:
            if attribute not in ranking_tokens:
                ranking_tokens.append(attribute)
            if attribute not in name_tokens:
                name_tokens.append(attribute)

        fetch_limit = public_limit
        maybe_rerank = bool(
            self.llm_config.rerank_enabled
            and self.llm_ranker
            and self.llm_ranker.available
        )
        if maybe_rerank:
            fetch_limit = min(
                max(self.llm_config.candidate_k, public_limit),
                50,
            )

        rows = self.repository.search_products(
            query=normalize_product_query(query),
            tokens=tokens,
            filters=metadata,
            ranking_tokens=ranking_tokens,
            rank_query=name_phrase or ranking_query_text(query),
            limit=fetch_limit,
            name_matching=self.name_matching_enabled,
            name_tokens=name_tokens,
            name_phrase=name_phrase,
            candidate_limit=self.name_candidate_limit,
        )
        db_candidate_count = len(rows)

        if (
            maybe_rerank
            and needs_rerank(query, metadata, tokens, db_candidate_count)
        ):
            reranked = self.llm_ranker.rerank(query, metadata, rows, usage)
            llm_rerank_used = reranked is not rows and usage.calls > 0
            rows = reranked
            final_limit = (
                min(public_limit, self.llm_config.final_k)
                if llm_rerank_used
                else public_limit
            )
        else:
            final_limit = public_limit

        rows = rows[:final_limit]
        items = [_to_item(row) for row in rows]
        debug = self._debug(
            query=query,
            metadata=metadata,
            deterministic=deterministic,
            started=started,
            usage=usage,
            llm_understanding_used=llm_understanding_used,
            llm_metadata_fallback_used=llm_metadata_fallback_used,
            metadata_confidence=confidence.score,
            metadata_confidence_reasons=list(confidence.reasons),
            llm_rerank_used=llm_rerank_used,
            db_candidate_count=db_candidate_count,
            llm_candidate_count=db_candidate_count if llm_rerank_used else 0,
            attributes=attributes,
            understanding_raw=understanding_raw,
            fallback_triggered=use_metadata_fallback,
            result_count=len(items),
        )
        return ProductSearchOutcome(items=items, debug=debug)

    def search_with_diagnostics(
        self,
        query: str,
        gold_codes: list[str] | None = None,
        ranked_limit: int = 500,
    ) -> dict:
        query = query.strip()[:200]
        metadata = extract_product_metadata(
            query,
            allow_negatives=self.negative_constraints_enabled,
        )
        tokens = search_tokens_for_metadata(query, metadata)
        confidence = assess_metadata_confidence(
            query,
            metadata,
            tokens,
            threshold=self.llm_config.metadata_confidence_threshold,
        )
        if (
            self.llm_config.metadata_fallback_enabled
            and self.llm_ranker
            and confidence.needs_llm
        ):
            metadata = self.llm_ranker.understand_metadata(query, metadata).metadata
            tokens = search_tokens_for_metadata(query, metadata)
        ranking_tokens = ranking_tokens_for_query(query)
        name_tokens = product_name_tokens(query, metadata)
        name_phrase = product_name_phrase(query, metadata)
        rank_query = name_phrase or ranking_query_text(query)
        common = dict(
            query=normalize_product_query(query),
            tokens=tokens,
            filters=metadata,
            ranking_tokens=ranking_tokens,
            rank_query=rank_query,
            name_matching=self.name_matching_enabled,
            name_tokens=name_tokens,
            name_phrase=name_phrase,
        )
        filtered = self.repository.inspect_filtered_candidates(
            **common,
            gold_codes=gold_codes or [],
        )
        ranked: list[dict] = []
        if tokens or metadata.has_filters():
            ranked = self.repository.search_products(
                **common,
                limit=min(max(ranked_limit, 1), 500),
                candidate_limit=self.name_candidate_limit,
                max_fetch=500,
            )
        ranked_codes = [str(row.get("product_code") or "") for row in ranked]
        ranked_codes = [code for code in ranked_codes if code]
        return {
            "query": query,
            "normalized_query": normalize_product_query(query),
            "metadata": metadata.as_dict(),
            "search_tokens": tokens,
            "ranking_tokens": ranking_tokens,
            "name_tokens": name_tokens,
            "name_phrase": name_phrase,
            "name_matching_enabled": self.name_matching_enabled,
            "negative_constraints_enabled": self.negative_constraints_enabled,
            "apply_token_filters": filtered["apply_token_filters"],
            "token_filters": filtered["token_filters"],
            "hard_sql_filters": filtered["where_sql"],
            "hard_filtered_count": filtered["hard_filtered_count"],
            "restrict_name_pool": filtered["restrict_name_pool"],
            "gold_in_filtered": filtered["gold_in_filtered"],
            "ranked_codes": ranked_codes,
            "ranked_preview": [
                {
                    "rank": index,
                    "product_code": row.get("product_code"),
                    "product_name": row.get("product_name"),
                    "brand": row.get("brand"),
                    "type": row.get("type"),
                    "last_status": row.get("last_status"),
                }
                for index, row in enumerate(ranked[:20], start=1)
            ],
            "top_3": ranked_codes[:3],
            "top_5": ranked_codes[:5],
            "top_10": ranked_codes[:10],
        }

    def _debug(
        self,
        query: str,
        metadata: ProductSearchMetadata,
        deterministic: ProductSearchMetadata,
        started: float,
        usage: LlmUsage,
        llm_understanding_used: bool = False,
        llm_metadata_fallback_used: bool = False,
        metadata_confidence: float = 1.0,
        metadata_confidence_reasons: list[str] | None = None,
        llm_rerank_used: bool = False,
        db_candidate_count: int = 0,
        llm_candidate_count: int = 0,
        attributes: list[str] | None = None,
        understanding_raw: dict | None = None,
        fallback_triggered: bool = False,
        result_count: int = 0,
    ) -> ProductSearchDebug:
        debug = ProductSearchDebug(
            query=query,
            normalized_query=normalize_product_query(query),
            deterministic_metadata=deterministic.as_dict(),
            metadata=metadata.as_dict(),
            llm_query_understanding_used=llm_understanding_used,
            llm_metadata_fallback_used=llm_metadata_fallback_used,
            metadata_confidence=metadata_confidence,
            metadata_confidence_reasons=metadata_confidence_reasons or [],
            llm_rerank_used=llm_rerank_used,
            db_candidate_count=db_candidate_count,
            llm_candidate_count=llm_candidate_count,
            llm_calls=usage.calls,
            llm_input_tokens=usage.input_tokens,
            llm_output_tokens=usage.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            fallback_reason=usage.fallback_reason,
            llm_understanding=understanding_raw,
            attributes=attributes or [],
        )
        _log_product_search(query, debug, fallback_triggered, result_count)
        return debug


def _metadata_source(debug: ProductSearchDebug, fallback_triggered: bool) -> str:
    if debug.llm_metadata_fallback_used:
        return "llm_fallback"
    if fallback_triggered:
        return "fallback_failed"
    return "deterministic"


def _hard_filters(metadata: dict) -> dict:
    applied = {}
    for key, value in metadata.items():
        if value in (None, "", [], False):
            continue
        applied[key] = value
    return applied


def _log_product_search(
    query: str,
    debug: ProductSearchDebug,
    fallback_triggered: bool,
    result_count: int,
) -> None:
    llm_failed = bool(debug.fallback_reason)
    llm_success = bool(debug.llm_metadata_fallback_used) and not llm_failed
    metadata_source = _metadata_source(debug, fallback_triggered)
    payload = {
        "event": "product_search_completed",
        "metadata_source": metadata_source,
        "metadata_confidence": debug.metadata_confidence,
        "metadata_confidence_reasons": debug.metadata_confidence_reasons,
        "llm_fallback_triggered": fallback_triggered,
        "llm_fallback_used": debug.llm_metadata_fallback_used,
        "llm_success": llm_success,
        "llm_failure": llm_failed,
        "fallback_reason": debug.fallback_reason,
        "candidate_count": debug.db_candidate_count,
        "result_count": result_count,
        "empty": result_count == 0,
        "hard_filters": _hard_filters(debug.metadata),
        "latency_ms": round(debug.latency_ms, 3),
    }
    payload.update(safe_query_ref(query))
    log_event(**payload)


def _to_item(row: dict) -> ProductSearchItem:
    return ProductSearchItem(
        product_code=row.get("product_code"),
        product_name=row.get("product_name"),
        brand=row.get("brand"),
        type=row.get("type"),
        sku=row.get("sku"),
        category_level1=row.get("category_level1"),
        category_level2=row.get("category_level2"),
        last_status=row.get("last_status"),
        available_qty=row.get("available_qty"),
        special_price=row.get("special_price"),
        color=row.get("color"),
        size=row.get("size"),
    )
