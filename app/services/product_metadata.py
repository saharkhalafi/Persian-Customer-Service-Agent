from __future__ import annotations

import functools
import re
from dataclasses import asdict, dataclass

from app.services.product_brand_aliases import (
    brand_aliases_longest_first,
    canonicalize_brand,
)
from app.services.product_name_matching import unique_keep_order
from app.services.product_query_normalizer import (
    _strip_zwnj,
    catalog_search_tokens,
    extract_search_tokens,
    fold_search_text,
    normalize_product_query,
)

_PRODUCT_PHRASE_EXTENDERS = frozenset(
    {
        "ماسک",
        "تثبیت",
        "ناخن",
        "ویتامین",
    }
)


_FA_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_WORD_NUMBERS = {
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "ده": 10,
    "بیست": 20,
    "صد": 100,
}

# longest catalog types first
_CATEGORY_ALIASES: tuple[tuple[str, str], ...] = (
    ("کیف لوازم آرایشی", "کیف لوازم آرایشی"),
    ("رژ لب مایع", "رژ لب مایع"),
    ("رژ لب جامد", "رژ لب جامد"),
    ("مام ضد تعریق", "مام"),
    ("کرم ضد آفتاب", "کرم ضدآفتاب"),
    ("کرم ضدآفتاب", "کرم ضدآفتاب"),
    ("ضدآفتاب", "کرم ضدآفتاب"),
    ("عطر جیبی", "عطر جیبی"),
    ("عطر زنانه", "عطر"),
    ("لاک ناخن", "لاک ناخن"),
    ("خط چشم", "خط چشم"),
    ("ماسک مو", "ماسک مو"),
    ("دستمال مرطوب", "دستمال مرطوب"),
    ("ماشین اصلاح", "ماشین اصلاح"),
    ("شوینده صورت", "شوینده صورت"),
    ("کرم دست", "کرم دست"),
    ("رنگ مو", "رنگ مو"),
    ("رژگونه", "رژگونه"),
    ("ماساژور", "ماساژور"),
    ("پنکک", "پنکک"),
    ("مسواک", "مسواک"),
    ("کانسیلر", "کانسیلر"),
)

_COLORS = (
    "مشکی",
    "سفید",
    "طوسی",
    "خاکستری",
    "قرمز",
    "آبی",
    "سبز",
    "صورتی",
    "طلایی",
    "نقره ای",
    "بنفش",
    "قهوه ای",
    "نارنجی",
)

_MATERIALS = (
    "چرم",
    "کتان",
    "پنبه",
    "ابریشم",
    "پلاستیک",
    "فلزی",
)

_GENDERS = (
    "زنانه",
    "مردانه",
    "دخترانه",
    "پسرانه",
)

_PRICE_TOKEN_PATTERN = re.compile(
    r"^[\d۰-۹]+(هزارتومان|هزارتومن|هزار|میلیون|تومان|تومن)?$",
    flags=re.UNICODE,
)

_NEGATIVE_BRAND_PREFIXES = (
    r"بدون\s+برند",
    r"به\s+جز\s+برند",
    r"غیر\s+از\s+برند",
    r"بدون",
    r"به\s+جز",
    r"غیر\s+از",
)


@dataclass(frozen=True)
class ProductSearchMetadata:
    brand: str | None = None
    category: str | None = None
    color: str | None = None
    material: str | None = None
    gender: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    availability: bool | None = None
    exclude_brands: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)

    def has_filters(self) -> bool:
        return self.has_positive_filters() or bool(self.exclude_brands)

    def has_positive_filters(self) -> bool:
        return any(
            (
                self.brand,
                self.category,
                self.color,
                self.material,
                self.gender,
                self.price_min is not None,
                self.price_max is not None,
                self.availability is not None,
            )
        )


def _normalize_text(query: str) -> str:
    return re.sub(r"\s+", " ", _strip_zwnj(str(query or "")).strip())


def _parse_number_token(token: str) -> float | None:
    if token in _WORD_NUMBERS:
        return float(_WORD_NUMBERS[token])
    digits = token.translate(_FA_DIGITS)
    if digits.isdigit():
        return float(digits)
    return None


def _amount_from_parts(num_token: str, scale: str | None) -> float | None:
    value = _parse_number_token(num_token)
    if value is None:
        return None
    if scale in {"هزار", "هزارتومان", "هزارتومن"}:
        return value * 1000
    if scale == "میلیون":
        return value * 1_000_000
    return value


def _extract_brand(query: str) -> str | None:
    text = _normalize_text(query).casefold()
    for alias, canonical in brand_aliases_longest_first():
        if re.search(rf"(^|\s){re.escape(alias)}(\s|$)", text):
            return canonical
    return None


def _extract_phrase(
    query: str,
    phrases: tuple[tuple[str, str], ...] | tuple[str, ...],
) -> str | None:
    text = _normalize_text(query)
    folded = text.casefold()
    items: list[tuple[str, str]]
    if phrases and isinstance(phrases[0], tuple):
        items = list(phrases)  # type: ignore[arg-type]
    else:
        items = [(phrase, phrase) for phrase in phrases]  # type: ignore[misc]
    for alias, canonical in sorted(items, key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"(^|\s){re.escape(alias)}(\s|$)", folded) or re.search(
            rf"(^|\s){re.escape(alias)}(\s|$)", text
        ):
            return canonical
    return None


def _extract_price(query: str) -> tuple[float | None, float | None]:
    text = _normalize_text(query)
    text = text.translate(_FA_DIGITS)
    text = text.replace(" ", "")
    # keep words needed for bounds/scales
    spaced = _normalize_text(query).translate(_FA_DIGITS)

    between = re.search(
        r"بین\s*([\d۰-۹]+|یک|دو|سه|ده|بیست|صد)\s*(هزار|میلیون)?"
        r"\s*تا\s*([\d۰-۹]+|یک|دو|سه|ده|بیست|صد)\s*(هزار|میلیون)?",
        spaced,
    )
    if between:
        low = _amount_from_parts(between.group(1), between.group(2) or between.group(4))
        high = _amount_from_parts(between.group(3), between.group(4) or between.group(2))
        if low is not None and high is not None:
            return low, high

    compact = re.sub(r"\s+", "", spaced)
    compact_match = re.search(
        r"(زیر|بالای|بالا)?(\d+)(هزارتومان|هزارتومن|هزار|میلیون)?",
        compact,
    )
    spaced_match = re.search(
        r"(زیر|بالای|بالا)\s+([\d]+|یک|دو|سه|ده|بیست|صد)\s*(هزارتومان|هزارتومن|هزار|میلیون)?",
        spaced,
    )
    match = spaced_match or compact_match
    if not match:
        return None, None

    bound = match.group(1)
    amount = _amount_from_parts(match.group(2), match.group(3))
    if amount is None:
        return None, None
    if bound in {"بالای", "بالا"}:
        return amount, None
    if bound == "زیر":
        return None, amount
    return None, None


def _extract_availability(query: str) -> bool | None:
    text = _normalize_text(query)
    if re.search(r"ناموجود|موجود نیست", text):
        return False
    if re.search(r"موجوده|موجود\b", text):
        return True
    return None


def canonicalize_category(value: str | None) -> str | None:
    text = _normalize_text(value or "")
    if not text:
        return None
    folded = text.casefold()
    for alias, canonical in sorted(
        _CATEGORY_ALIASES, key=lambda item: len(item[0]), reverse=True
    ):
        if folded == alias.casefold() or folded == canonical.casefold():
            return canonical
    return None


def canonicalize_color(value: str | None) -> str | None:
    return _extract_phrase(value or "", _COLORS)


def canonicalize_material(value: str | None) -> str | None:
    return _extract_phrase(value or "", _MATERIALS)


def canonicalize_gender(value: str | None) -> str | None:
    return _extract_phrase(value or "", _GENDERS)


def peek_raw_category(query: str) -> str | None:
    return _extract_phrase(query, _CATEGORY_ALIASES)


def matching_categories(query: str) -> tuple[str, ...]:
    text = _normalize_text(query)
    folded = text.casefold()
    found: list[str] = []
    for alias, canonical in _CATEGORY_ALIASES:
        if re.search(rf"(^|\s){re.escape(alias)}(\s|$)", folded) or re.search(
            rf"(^|\s){re.escape(alias)}(\s|$)", text
        ):
            if canonical not in found:
                found.append(canonical)
    return tuple(found)


def known_category_values() -> tuple[str, ...]:
    values: list[str] = []
    for alias, canonical in _CATEGORY_ALIASES:
        if canonical not in values:
            values.append(canonical)
        if alias not in values:
            values.append(alias)
    return tuple(values)


def known_color_values() -> tuple[str, ...]:
    return _COLORS


def known_material_values() -> tuple[str, ...]:
    return _MATERIALS


def known_gender_values() -> tuple[str, ...]:
    return _GENDERS


def query_has_numeric_price(query: str) -> bool:
    price_min, price_max = _extract_price(query)
    return price_min is not None or price_max is not None


def extract_excluded_brands(query: str) -> tuple[tuple[str, ...], str]:
    remaining = fold_search_text(query)
    if not remaining:
        return (), remaining

    excluded: list[str] = []
    for alias, canonical in brand_aliases_longest_first():
        if canonical in excluded:
            continue
        prefix = "|".join(_NEGATIVE_BRAND_PREFIXES)
        patterns = (
            rf"(^|\s)(?:{prefix})\s+{re.escape(alias)}(\s|$)",
            rf"(^|\s)از\s+(?:برند\s+)?{re.escape(alias)}\s+نه(\s|$)",
        )
        for pattern in patterns:
            matched = re.search(pattern, remaining, flags=re.IGNORECASE)
            if not matched:
                continue
            excluded.append(canonical)
            remaining = re.sub(pattern, r"\1\2", remaining, flags=re.IGNORECASE)
            remaining = re.sub(r"\s+", " ", remaining).strip()
            break
    return tuple(unique_keep_order(excluded)), remaining


def category_is_embedded_in_product_phrase(
    query: str,
    metadata: ProductSearchMetadata,
) -> bool:
    if not metadata.category:
        return False
    leftover = search_tokens_for_metadata(query, metadata)
    leftover_keys = {fold_search_text(token).casefold() for token in leftover}
    if leftover_keys & {item.casefold() for item in _PRODUCT_PHRASE_EXTENDERS}:
        return True
    text = fold_search_text(query)
    category = fold_search_text(metadata.category)
    if category and re.search(rf"{re.escape(category)}\s+و\s+\S+", text):
        return True
    if category == "رنگ مو" and re.search(r"(^|\s)ماسک(\s|$)", text):
        return True
    return False


def extract_product_metadata(
    query: str,
    allow_negatives: bool = True,
) -> ProductSearchMetadata:
    if not str(query or "").strip():
        return ProductSearchMetadata()
    return _extract_product_metadata_cached(query, allow_negatives)


@functools.lru_cache(maxsize=512)
def _extract_product_metadata_cached(
    query: str,
    allow_negatives: bool,
) -> ProductSearchMetadata:
    working = query
    exclude_brands: tuple[str, ...] = ()
    if allow_negatives:
        exclude_brands, stripped = extract_excluded_brands(query)
        if exclude_brands:
            working = stripped

    brand = _extract_brand(working)
    if brand and brand in exclude_brands:
        brand = None
    category = _extract_phrase(working, _CATEGORY_ALIASES)
    color = _extract_phrase(working, _COLORS)
    material = _extract_phrase(working, _MATERIALS)
    gender = _extract_phrase(working, _GENDERS)
    price_min, price_max = _extract_price(working)
    availability = _extract_availability(working)
    tentative = ProductSearchMetadata(
        brand=brand,
        category=category,
        color=color,
        material=material,
        gender=gender,
        price_min=price_min,
        price_max=price_max,
        availability=availability,
        exclude_brands=exclude_brands,
    )
    if category and category_is_embedded_in_product_phrase(working, tentative):
        category = None

    return ProductSearchMetadata(
        brand=brand,
        category=category,
        color=color,
        material=material,
        gender=gender,
        price_min=price_min,
        price_max=price_max,
        availability=availability,
        exclude_brands=exclude_brands,
    )


def metadata_consumed_tokens(metadata: ProductSearchMetadata) -> set[str]:
    consumed: set[str] = set()
    if metadata.brand:
        canonical = canonicalize_brand(metadata.brand) or metadata.brand
        for alias, mapped in brand_aliases_longest_first():
            if mapped.casefold() == canonical.casefold():
                consumed.update(alias.split())
                consumed.add(alias)
        consumed.update(canonical.casefold().split())
    if metadata.category:
        for alias, canonical in _CATEGORY_ALIASES:
            if canonical.casefold() == metadata.category.casefold():
                consumed.update(alias.casefold().split())
                consumed.add(alias.casefold())
        consumed.update(metadata.category.casefold().split())
    for value in (metadata.color, metadata.material, metadata.gender):
        if value:
            consumed.update(value.casefold().split())
            consumed.add(value.casefold())
    return consumed


def _is_price_token(token: str) -> bool:
    compact = _strip_zwnj(token).translate(_FA_DIGITS).replace(" ", "")
    if compact in {
        "هزار",
        "میلیون",
        "تومان",
        "تومن",
        "هزارتومان",
        "هزارتومن",
        "زیر",
        "بالای",
        "بالا",
        "بین",
    }:
        return True
    return bool(_PRICE_TOKEN_PATTERN.fullmatch(compact))


def search_tokens_for_metadata(
    query: str,
    metadata: ProductSearchMetadata,
) -> list[str]:
    if not metadata.has_filters():
        return catalog_search_tokens(query)

    tokens = catalog_search_tokens(query)
    consumed = metadata_consumed_tokens(metadata)
    leftover = [
        token
        for token in tokens
        if token.casefold() not in consumed
        and _strip_zwnj(token).casefold() not in consumed
    ]
    if metadata.price_min is not None or metadata.price_max is not None:
        leftover = [token for token in leftover if not _is_price_token(token)]
    return leftover


def ranking_tokens_for_query(query: str) -> list[str]:
    return extract_search_tokens(query)


def ranking_query_text(query: str) -> str:
    return normalize_product_query(query)


def product_name_tokens(
    query: str,
    metadata: ProductSearchMetadata | None = None,
) -> list[str]:
    tokens = extract_search_tokens(query)
    kept = [fold_search_text(token) for token in tokens if token and not _is_price_token(token)]
    if metadata and (metadata.price_min is not None or metadata.price_max is not None):
        kept = [token for token in kept if not _is_price_token(token)]
    return unique_keep_order(kept)[:12]


def product_name_phrase(
    query: str,
    metadata: ProductSearchMetadata | None = None,
) -> str:
    return " ".join(product_name_tokens(query, metadata))
