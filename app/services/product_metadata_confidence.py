from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.product_brand_aliases import BRAND_ALIAS_MAP, canonicalize_brand
from app.services.product_metadata import (
    ProductSearchMetadata,
    category_is_embedded_in_product_phrase,
    matching_categories,
    peek_raw_category,
)
from app.services.product_query_normalizer import fold_search_text, is_soft_filter_token


UNKNOWN_BRAND_ALIAS = "unknown_or_near_miss_brand"
AMBIGUOUS_CATEGORY = "ambiguous_category"
CONFLICTING_METADATA = "conflicting_metadata"
PARTIAL_PRODUCT_PHRASE = "partial_product_phrase"
COLLOQUIAL_NO_METADATA = "colloquial_without_reliable_metadata"
IMPLICIT_USE_NO_CATEGORY = "implicit_use_without_category"

REASON_PENALTY = 0.35

_COLLOQUIAL = re.compile(
    r"یه چیزی|یه دونه|یه مدل|نمیخوام خیلی|گرون نباشه|خوب باشه|"
    r"چی پیشنهاد|پیشنهاد میدی|نمی دونم|نمیدونم",
    flags=re.UNICODE,
)
_IMPLICIT_USE = re.compile(
    r"برای\s+\S+|مناسب\s+\S+|موهای|گردن|صورت|خشک|چرب|حساس|مرطوب",
    flags=re.UNICODE,
)
_PRODUCT_PHRASE_EXTENDERS = frozenset({"ماسک", "تثبیت", "ناخن", "ویتامین"})


@dataclass(frozen=True)
class MetadataConfidenceAssessment:
    score: float
    reasons: tuple[str, ...]
    locked_fields: tuple[str, ...]
    needs_llm: bool

    def as_dict(self) -> dict:
        return {
            "score": self.score,
            "reasons": list(self.reasons),
            "locked_fields": list(self.locked_fields),
            "needs_llm": self.needs_llm,
        }


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > 2:
        return 99
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def possible_unknown_brand_alias(tokens: list[str], extracted_brand: str | None) -> str | None:
    if extracted_brand:
        return None
    aliases = [alias for alias in BRAND_ALIAS_MAP if len(alias) >= 4]
    for token in tokens:
        folded = fold_search_text(token).casefold()
        if len(folded) < 4 or is_soft_filter_token(token):
            continue
        if canonicalize_brand(token):
            continue
        for alias in aliases:
            if abs(len(folded) - len(alias)) > 1:
                continue
            if _edit_distance(folded, alias) <= 1:
                return token
    return None


def _locked_fields(metadata: ProductSearchMetadata) -> tuple[str, ...]:
    locked: list[str] = []
    if metadata.brand:
        locked.append("brand")
    if metadata.category:
        locked.append("category")
    if metadata.color:
        locked.append("color")
    if metadata.material:
        locked.append("material")
    if metadata.gender:
        locked.append("gender")
    if metadata.price_min is not None:
        locked.append("price_min")
    if metadata.price_max is not None:
        locked.append("price_max")
    if metadata.availability is not None:
        locked.append("availability")
    if metadata.exclude_brands:
        locked.append("exclude_brands")
    return tuple(locked)


def assess_metadata_confidence(
    query: str,
    metadata: ProductSearchMetadata,
    tokens: list[str],
    threshold: float = 0.75,
) -> MetadataConfidenceAssessment:
    reasons: list[str] = []
    leftover = [token for token in tokens if token and not is_soft_filter_token(token)]

    raw_category = peek_raw_category(query)
    if raw_category and not metadata.category:
        tentative = ProductSearchMetadata(
            brand=metadata.brand,
            category=raw_category,
            color=metadata.color,
            material=metadata.material,
            gender=metadata.gender,
            price_min=metadata.price_min,
            price_max=metadata.price_max,
            availability=metadata.availability,
            exclude_brands=metadata.exclude_brands,
        )
        if category_is_embedded_in_product_phrase(query, tentative):
            reasons.append(AMBIGUOUS_CATEGORY)

    categories = matching_categories(query)
    if len(set(categories)) > 1:
        reasons.append(CONFLICTING_METADATA)

    if possible_unknown_brand_alias(leftover, metadata.brand):
        reasons.append(UNKNOWN_BRAND_ALIAS)

    leftover_keys = {fold_search_text(token).casefold() for token in leftover}
    if (
        not metadata.category
        and leftover_keys & {item.casefold() for item in _PRODUCT_PHRASE_EXTENDERS}
    ):
        reasons.append(PARTIAL_PRODUCT_PHRASE)

    if not metadata.has_positive_filters():
        if _COLLOQUIAL.search(query or ""):
            reasons.append(COLLOQUIAL_NO_METADATA)
        elif _IMPLICIT_USE.search(query or ""):
            reasons.append(IMPLICIT_USE_NO_CATEGORY)

    unique_reasons = tuple(dict.fromkeys(reasons))
    score = max(0.0, 1.0 - REASON_PENALTY * len(unique_reasons))
    return MetadataConfidenceAssessment(
        score=score,
        reasons=unique_reasons,
        locked_fields=_locked_fields(metadata),
        needs_llm=score < threshold and bool(unique_reasons),
    )
