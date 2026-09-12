from __future__ import annotations

import os
import re

from app.services.product_brand_aliases import (
    brand_aliases_longest_first,
    brand_match_keys,
    canonicalize_brand,
)
from app.services.product_query_normalizer import fold_search_text


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def load_name_matching_enabled(default: bool = False) -> bool:
    return _as_bool(os.getenv("PRODUCT_NAME_MATCHING_ENABLED"), default)


def load_negative_constraints_enabled(default: bool = True) -> bool:
    return _as_bool(os.getenv("PRODUCT_NEGATIVE_CONSTRAINTS_ENABLED"), default)


def load_name_candidate_limit(default: int = 80) -> int:
    return _as_int(os.getenv("PRODUCT_NAME_CANDIDATE_LIMIT"), default, 10, 200)


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for value in values:
        text = fold_search_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def name_token_variants(token: str) -> list[str]:
    folded = fold_search_text(token)
    variants = [folded]
    canonical = canonicalize_brand(folded)
    if canonical:
        variants.extend(brand_match_keys(canonical))
    return unique_keep_order(variants)[:6]


def name_phrase_variants(phrase: str) -> list[str]:
    folded = fold_search_text(phrase)
    if not folded:
        return []
    variants = [folded]
    working = folded.casefold()
    for alias, canonical in brand_aliases_longest_first():
        pattern = rf"(^|\s){re.escape(alias)}(\s|$)"
        if not re.search(pattern, working):
            continue
        for replacement in unique_keep_order([canonical, *brand_match_keys(canonical)])[:3]:
            replaced = re.sub(pattern, rf"\1{replacement}\2", working)
            variants.append(replaced)
    return unique_keep_order(variants)[:6]
