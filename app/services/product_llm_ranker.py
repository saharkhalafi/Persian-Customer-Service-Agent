from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable

from app.core.observability import llm_purpose_scope
from app.services.product_brand_aliases import (
    CANONICAL_BRAND_ALIASES,
    canonicalize_brand,
)
from app.services.product_llm_config import ProductLlmConfig, load_product_llm_config
from app.services.product_metadata import (
    ProductSearchMetadata,
    canonicalize_category,
    canonicalize_color,
    canonicalize_gender,
    canonicalize_material,
    category_is_embedded_in_product_phrase,
    known_category_values,
    known_color_values,
    known_gender_values,
    known_material_values,
    peek_raw_category,
    query_has_numeric_price,
)


GenerateJsonFn = Callable[[str, dict[str, Any], float], "LlmJsonResult"]

_COLLOQUIAL = re.compile(
    r"یه چیزی|یه دونه|یه مدل|نمیخوام خیلی|گرون نباشه|خوب باشه|"
    r"چی پیشنهاد|پیشنهاد میدی|نمی دونم|نمیدونم",
    flags=re.UNICODE,
)
_IMPLICIT_USE = re.compile(
    r"برای\s+\S+|مناسب\s+\S+|موهای|گردن|صورت|خشک|چرب|حساس",
    flags=re.UNICODE,
)
_IMPLICIT_PRICE = re.compile(
    r"گرون|گران|ارزان|ارزون|بودجه|قیمت مناسب",
    flags=re.UNICODE,
)
_QUALITY = re.compile(r"خوب|بهترین|بهترین|بهتر\b", flags=re.UNICODE)
_COMPLEX_INTENT = re.compile(
    r"ولی|اما|همچنین|هم\s+|نه خیلی|ترجیح",
    flags=re.UNICODE,
)


@dataclass
class LlmJsonResult:
    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LlmUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    fallback_reason: str | None = None

    def add(self, result: LlmJsonResult) -> None:
        self.calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens


@dataclass
class QueryUnderstanding:
    metadata: ProductSearchMetadata
    attributes: list[str] = field(default_factory=list)
    price_intent: str | None = None
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


UNDERSTAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category_or_type": {"type": "string", "nullable": True},
        "brand": {"type": "string", "nullable": True},
        "color": {"type": "string", "nullable": True},
        "material": {"type": "string", "nullable": True},
        "gender": {"type": "string", "nullable": True},
        "attributes": {"type": "array", "items": {"type": "string"}},
        "price_intent": {
            "type": "string",
            "enum": ["budget", "premium", "range", "none"],
        },
        "price_min": {"type": "number", "nullable": True},
        "price_max": {"type": "number", "nullable": True},
        "availability": {"type": "boolean", "nullable": True},
        "confidence": {"type": "number"},
    },
    "required": ["confidence", "attributes", "price_intent"],
}

METADATA_FALLBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "brand": {"type": "string", "nullable": True},
        "category_or_type": {"type": "string", "nullable": True},
        "color": {"type": "string", "nullable": True},
        "material": {"type": "string", "nullable": True},
        "gender": {"type": "string", "nullable": True},
        "price_min": {"type": "number", "nullable": True},
        "price_max": {"type": "number", "nullable": True},
        "confidence": {"type": "number"},
    },
    "required": ["confidence"],
}

RERANK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "product_code": {"type": "string"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["product_code", "score", "reason"],
            },
        }
    },
    "required": ["results"],
}


def _has_hard_filters(metadata: ProductSearchMetadata) -> bool:
    return any(
        (
            metadata.brand,
            metadata.category,
            metadata.color,
            metadata.material,
            metadata.gender,
            metadata.price_min is not None,
            metadata.price_max is not None,
            metadata.availability is not None,
        )
    )


def _has_soft_language(query: str) -> bool:
    return bool(
        _COLLOQUIAL.search(query)
        or _IMPLICIT_USE.search(query)
        or _IMPLICIT_PRICE.search(query)
        or _QUALITY.search(query)
        or _COMPLEX_INTENT.search(query)
    )


def needs_query_understanding(
    query: str,
    metadata: ProductSearchMetadata,
    tokens: list[str],
) -> bool:
    implicit_price = bool(_IMPLICIT_PRICE.search(query)) and metadata.price_min is None and metadata.price_max is None
    implicit_use_without_type = bool(_IMPLICIT_USE.search(query)) and not metadata.category
    colloquial = bool(_COLLOQUIAL.search(query))
    incomplete = not _has_hard_filters(metadata) and len(tokens) <= 1

    if implicit_price or implicit_use_without_type or colloquial:
        return True
    if incomplete and _has_soft_language(query):
        return True
    return False


def needs_rerank(
    query: str,
    metadata: ProductSearchMetadata,
    tokens: list[str],
    candidate_count: int,
) -> bool:
    if candidate_count <= 1:
        return False
    if _COLLOQUIAL.search(query) or _QUALITY.search(query) or _IMPLICIT_USE.search(query):
        return True
    if _COMPLEX_INTENT.search(query):
        return True
    if tokens and not metadata.brand and not metadata.category:
        return True
    return False


def merge_llm_understanding(
    deterministic: ProductSearchMetadata,
    llm_payload: dict[str, Any],
    query: str,
    confidence_threshold: float,
) -> QueryUnderstanding:
    try:
        confidence = float(llm_payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    attributes = [
        str(item).strip()
        for item in (llm_payload.get("attributes") or [])
        if str(item).strip()
    ]
    price_intent = llm_payload.get("price_intent")
    if price_intent not in {"budget", "premium", "range"}:
        price_intent = None

    if confidence < confidence_threshold:
        return QueryUnderstanding(
            metadata=deterministic,
            attributes=attributes,
            price_intent=price_intent,
            confidence=confidence,
            raw=llm_payload,
        )

    brand = deterministic.brand or canonicalize_brand(llm_payload.get("brand"))
    category = deterministic.category or canonicalize_category(
        llm_payload.get("category_or_type")
    )
    if not deterministic.category and category:
        suppressed = peek_raw_category(query)
        if suppressed and category == suppressed:
            tentative = ProductSearchMetadata(
                brand=deterministic.brand,
                category=suppressed,
                color=deterministic.color,
                material=deterministic.material,
                gender=deterministic.gender,
                price_min=deterministic.price_min,
                price_max=deterministic.price_max,
                availability=deterministic.availability,
                exclude_brands=deterministic.exclude_brands,
            )
            if category_is_embedded_in_product_phrase(query, tentative):
                category = None
    color = deterministic.color or canonicalize_color(llm_payload.get("color"))
    material = deterministic.material or canonicalize_material(
        llm_payload.get("material")
    )
    gender = deterministic.gender or canonicalize_gender(llm_payload.get("gender"))

    price_min = deterministic.price_min
    price_max = deterministic.price_max
    if query_has_numeric_price(query):
        if price_min is None:
            price_min = _safe_float(llm_payload.get("price_min"))
        if price_max is None:
            price_max = _safe_float(llm_payload.get("price_max"))

    availability = deterministic.availability
    if availability is None and re.search(r"ناموجود|موجود", query or ""):
        raw_availability = llm_payload.get("availability")
        if isinstance(raw_availability, bool):
            availability = raw_availability

    return QueryUnderstanding(
        metadata=ProductSearchMetadata(
            brand=brand,
            category=category,
            color=color,
            material=material,
            gender=gender,
            price_min=price_min,
            price_max=price_max,
            availability=availability,
            exclude_brands=deterministic.exclude_brands,
        ),
        attributes=attributes,
        price_intent=price_intent,
        confidence=confidence,
        raw=llm_payload,
    )


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    price = row.get("sum_of_price")
    if price is None:
        price = row.get("special_price")
    return {
        "product_code": str(row.get("product_code") or ""),
        "product_name": row.get("product_name"),
        "brand": row.get("brand"),
        "category": row.get("type") or row.get("category_level1"),
        "color": row.get("color"),
        "material": row.get("material"),
        "price": price,
    }


def apply_rerank_order(
    candidates: list[dict[str, Any]],
    llm_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_code = {
        str(row.get("product_code") or ""): row
        for row in candidates
        if row.get("product_code")
    }
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in llm_results:
        code = str(item.get("product_code") or "")
        if code in by_code and code not in seen:
            ordered.append(by_code[code])
            seen.add(code)
    for row in candidates:
        code = str(row.get("product_code") or "")
        if code and code not in seen:
            ordered.append(row)
            seen.add(code)
    return ordered


class ProductLlmRanker:
    def __init__(
        self,
        generate_json: GenerateJsonFn | None = None,
        config: ProductLlmConfig | None = None,
    ):
        self._generate_json = generate_json
        self.config = config or load_product_llm_config()

    @property
    def available(self) -> bool:
        return self._generate_json is not None

    def understand_query(
        self,
        query: str,
        deterministic: ProductSearchMetadata,
        usage: LlmUsage | None = None,
    ) -> QueryUnderstanding:
        usage = usage or LlmUsage()
        if self._generate_json is None:
            usage.fallback_reason = "llm_unavailable"
            return QueryUnderstanding(metadata=deterministic, raw={})

        prompt = (
            "You extract structured product-search hints from a Persian shopper query.\n"
            "Return JSON only that matches the schema.\n"
            "Rules:\n"
            "- Do not invent exact catalog brands, categories, prices, or IDs.\n"
            "- If uncertain, use null and lower confidence.\n"
            "- Never convert vague price talk such as 'not too expensive' into a numeric price.\n"
            "- Use Persian attribute phrases when possible.\n"
            "- Only fill fields you can support from the query.\n"
            f"Query: {query}\n"
            f"Deterministic filters already extracted: {deterministic.as_dict()}\n"
        )
        try:
            with llm_purpose_scope("product_query_understanding"):
                result = self._generate_json(
                    prompt, UNDERSTAND_SCHEMA, self.config.timeout_seconds
                )
        except TimeoutError:
            usage.fallback_reason = "query_understanding_timeout"
            return QueryUnderstanding(metadata=deterministic, raw={})
        except Exception:
            usage.fallback_reason = "query_understanding_error"
            return QueryUnderstanding(metadata=deterministic, raw={})

        usage.add(result)
        if not isinstance(result.data, dict):
            usage.fallback_reason = "query_understanding_invalid_json"
            return QueryUnderstanding(metadata=deterministic, raw={})
        return merge_llm_understanding(
            deterministic,
            result.data,
            query,
            self.config.confidence_threshold,
        )

    def understand_metadata(
        self,
        query: str,
        deterministic: ProductSearchMetadata,
        usage: LlmUsage | None = None,
    ) -> QueryUnderstanding:
        usage = usage or LlmUsage()
        if self._generate_json is None:
            usage.fallback_reason = "llm_unavailable"
            return QueryUnderstanding(metadata=deterministic, raw={})

        brands = ", ".join(sorted(CANONICAL_BRAND_ALIASES))
        categories = ", ".join(known_category_values())
        colors = ", ".join(known_color_values())
        materials = ", ".join(known_material_values())
        genders = ", ".join(known_gender_values())
        prompt = (
            "Extract structured product-search metadata from a Persian shopper query.\n"
            "Return JSON only that matches the schema. Do not generate SQL.\n"
            "Rules:\n"
            "- Use only values from the allowed vocabularies below.\n"
            "- If a value is not in the allowed list, return null for that field.\n"
            "- Do not invent brands, categories, colors, materials, genders, or prices.\n"
            "- Do not copy leftover conversational words into metadata.\n"
            "- If uncertain, use null and lower confidence.\n"
            "- Never convert vague price talk into a numeric price.\n"
            f"Allowed brands: {brands}\n"
            f"Allowed categories: {categories}\n"
            f"Allowed colors: {colors}\n"
            f"Allowed materials: {materials}\n"
            f"Allowed genders: {genders}\n"
            f"Query: {query}\n"
            f"Deterministic filters already extracted: {deterministic.as_dict()}\n"
        )
        try:
            with llm_purpose_scope("product_metadata_fallback"):
                result = self._generate_json(
                    prompt, METADATA_FALLBACK_SCHEMA, self.config.timeout_seconds
                )
        except TimeoutError:
            usage.fallback_reason = "llm_metadata_timeout"
            return QueryUnderstanding(metadata=deterministic, raw={})
        except Exception:
            usage.fallback_reason = "llm_metadata_error"
            return QueryUnderstanding(metadata=deterministic, raw={})

        usage.add(result)
        if not isinstance(result.data, dict):
            usage.fallback_reason = "llm_metadata_invalid_json"
            return QueryUnderstanding(metadata=deterministic, raw={})
        return merge_llm_understanding(
            deterministic,
            result.data,
            query,
            self.config.confidence_threshold,
        )

    def rerank(
        self,
        query: str,
        metadata: ProductSearchMetadata,
        candidates: list[dict[str, Any]],
        usage: LlmUsage | None = None,
    ) -> list[dict[str, Any]]:
        usage = usage or LlmUsage()
        if self._generate_json is None or not candidates:
            usage.fallback_reason = usage.fallback_reason or "llm_unavailable"
            return candidates

        payload = [candidate_payload(row) for row in candidates]
        allowed = {item["product_code"] for item in payload if item["product_code"]}
        prompt = (
            "Rerank already-filtered product candidates for a Persian shopper query.\n"
            "Return JSON only.\n"
            "Rules:\n"
            "- Use only the provided product_code values.\n"
            "- Do not invent products, prices, brands, or codes.\n"
            "- Do not change filters; only reorder by relevance to soft preferences.\n"
            f"Query: {query}\n"
            f"Hard filters already applied: {metadata.as_dict()}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=False)}\n"
        )
        try:
            with llm_purpose_scope("product_rerank"):
                result = self._generate_json(
                    prompt, RERANK_SCHEMA, self.config.timeout_seconds
                )
        except TimeoutError:
            usage.fallback_reason = "rerank_timeout"
            return candidates
        except Exception:
            usage.fallback_reason = "rerank_error"
            return candidates

        usage.add(result)
        rows = result.data.get("results") if isinstance(result.data, dict) else None
        if not isinstance(rows, list):
            usage.fallback_reason = "rerank_invalid_json"
            return candidates

        valid = [
            item
            for item in rows
            if isinstance(item, dict)
            and str(item.get("product_code") or "") in allowed
        ]
        if not valid:
            usage.fallback_reason = "rerank_no_valid_codes"
            return candidates
        return apply_rerank_order(candidates, valid)


def gemini_generate_json(gemini_client: Any) -> GenerateJsonFn:
    def _call(prompt: str, schema: dict[str, Any], timeout_seconds: float) -> LlmJsonResult:
        def _invoke() -> LlmJsonResult:
            raw = gemini_client.generate_json(prompt, schema)
            return LlmJsonResult(
                data=getattr(raw, "data", {}),
                input_tokens=int(getattr(raw, "input_tokens", 0) or 0),
                output_tokens=int(getattr(raw, "output_tokens", 0) or 0),
            )

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_invoke)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeout as exc:
                future.cancel()
                raise TimeoutError("LLM request timed out") from exc

    return _call
