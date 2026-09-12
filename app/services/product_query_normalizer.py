from __future__ import annotations

import re


_PHRASE_FILLERS = (
    "نشونم بده",
    "نشانم بده",
    "بهم نشون بده",
    "برام نشون بده",
    "رو نشون بده",
    "رو نشان بده",
    "نشون بده",
    "نشان بده",
    "چی موجوده",
    "چی دارید",
    "چی دارین",
    "برای من",
    "به من",
    "می خواهم",
    "می خوام",
    "میخواستم",
    "می خواستم",
    "پیدا کن",
    "سفارش بدم",
    "چند تا",
)

_TOKEN_FILLERS = {
    "میخوام",
    "میخواهم",
    "میخواستم",
    "لطفا",
    "لطفاً",
    "برام",
    "بهم",
    "رو",
    "را",
    "محصولات",
    "محصول",
    "کالاها",
    "کالا",
    "نشون",
    "نشان",
    "بده",
    "بدهید",
    "بدین",
    "از",
    "چی",
    "چیز",
    "چیزی",
    "موجوده",
    "موجود",
    "دارید",
    "دارین",
    "داره",
    "دارم",
    "هست",
    "هستن",
    "باشه",
    "کن",
    "کنه",
    "کنید",
    "کنین",
    "میشه",
    "میتونم",
    "میتوانم",
    "یه",
    "این",
    "اون",
    "هم",
    "ولی",
    "اما",
    "که",
    "تو",
    "در",
    "برای",
    "من",
    "ما",
    "شما",
    "بخرم",
    "بخریم",
    "بدم",
    "ببینم",
    "ببین",
    "آیا",
    "اگر",
    "فقط",
    "همه",
    "اینا",
    "ممنون",
    "مرسی",
    "سلام",
    "برند",
    "های",
    "ها",
    "کدوم",
    "کدومش",
    "چندتا",
    "چند",
    "واسه",
    "کجاست",
    "چطوره",
    "چنده",
    "چیه",
    "چیست",
    "کیه",
    "بگو",
    "بگید",
    "بگین",
    "products",
    "product",
    "please",
    "show",
}

_PUNCT_PATTERN = re.compile(
    r"[؟?!،,.;:٪%()\[\]{}\"'«»…]+",
    flags=re.UNICODE,
)

_LETTER_FOLD = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
    }
)
# Never used as hard SQL AND tokens; remain ranking-only signals.
_SOFT_FILTER_TOKENS = {
    "کردن",
    "یا",
    "و",
    "حاوی",
    "مناسب",
    "خیلی",
    "خوب",
    "بهتر",
    "بهترین",
    "سی",
    "c",
}

_NON_CATALOG_CONSTRAINTS = {
    "زیر",
    "بالای",
    "بالا",
    "بین",
    "میلیون",
    "هزار",
    "تومن",
    "تومان",
    "صد",
    "ده",
    "بیست",
    "یک",
}

_DIGIT_CHARS = frozenset("0123456789۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩")

_TOKEN_PATTERN = re.compile(
    r"[^\w\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC"
    r"\u0660-\u0669\u06F0-\u06F9]+",
    flags=re.UNICODE,
)


def _strip_zwnj(value: str) -> str:
    return value.replace("\u200c", "")


def fold_search_text(value: str) -> str:
    text = _strip_zwnj(str(value or "")).translate(_LETTER_FOLD)
    text = _PUNCT_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_search_tokens(query: str) -> list[str]:
    text = fold_search_text(query)
    if not text:
        return []

    for phrase in sorted(_PHRASE_FILLERS, key=len, reverse=True):
        text = text.replace(phrase, " ")

    text = _TOKEN_PATTERN.sub(" ", text)
    tokens: list[str] = []
    seen: set[str] = set()

    for raw in text.split():
        token = raw.strip()
        if not token:
            continue
        if _strip_zwnj(token) in _TOKEN_FILLERS:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)

    return tokens


def _is_numeric_token(token: str) -> bool:
    return bool(token) and all(char in _DIGIT_CHARS for char in token)


def is_soft_filter_token(token: str) -> bool:
    return fold_search_text(token).casefold() in _SOFT_FILTER_TOKENS


def catalog_search_tokens(query: str) -> list[str]:
    tokens = extract_search_tokens(query)
    catalog_tokens = [
        token
        for token in tokens
        if token not in _NON_CATALOG_CONSTRAINTS and not _is_numeric_token(token)
    ]
    return catalog_tokens or tokens


def hard_catalog_tokens(tokens: list[str]) -> list[str]:
    return [
        token
        for token in tokens
        if token
        and token not in _NON_CATALOG_CONSTRAINTS
        and not _is_numeric_token(token)
        and not is_soft_filter_token(token)
    ]


def normalize_product_query(query: str) -> str:
    return " ".join(extract_search_tokens(query))
