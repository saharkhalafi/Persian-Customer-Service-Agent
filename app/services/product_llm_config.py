from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _as_float(value: str | None, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


@dataclass(frozen=True)
class ProductLlmConfig:
    query_understanding_enabled: bool = False
    rerank_enabled: bool = False
    metadata_fallback_enabled: bool = False
    candidate_k: int = 20
    final_k: int = 10
    confidence_threshold: float = 0.7
    metadata_confidence_threshold: float = 0.75
    timeout_seconds: float = 20.0
    model: str = "gemini-2.5-flash"

    def as_dict(self) -> dict:
        return asdict(self)


def load_product_llm_config() -> ProductLlmConfig:
    return ProductLlmConfig(
        query_understanding_enabled=_as_bool(
            os.getenv("LLM_QUERY_UNDERSTANDING_ENABLED")
        ),
        rerank_enabled=_as_bool(os.getenv("LLM_RERANK_ENABLED")),
        metadata_fallback_enabled=_as_bool(
            os.getenv("LLM_METADATA_FALLBACK_ENABLED"), True
        ),
        candidate_k=_as_int(os.getenv("LLM_RERANK_CANDIDATE_K"), 20, 1, 50),
        final_k=_as_int(os.getenv("LLM_FINAL_K"), 10, 1, 20),
        confidence_threshold=_as_float(
            os.getenv("LLM_CONFIDENCE_THRESHOLD"), 0.7, 0.0, 1.0
        ),
        metadata_confidence_threshold=_as_float(
            os.getenv("LLM_METADATA_CONFIDENCE_THRESHOLD"), 0.75, 0.0, 1.0
        ),
        timeout_seconds=_as_float(os.getenv("LLM_TIMEOUT_SECONDS"), 20.0, 1.0, 120.0),
        model=os.getenv("LLM_PRODUCT_MODEL") or "gemini-2.5-flash",
    )
