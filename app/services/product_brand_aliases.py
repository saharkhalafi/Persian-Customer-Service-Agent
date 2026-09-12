from __future__ import annotations


# canonical brand → accepted aliases (Persian, English, short forms)
# Add new brands here; comparison is always case-insensitive.
CANONICAL_BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "Salute": ("سالوته", "سالوت", "salute"),
    "Xiaomi": ("شیائومی", "xiaomi"),
    "Note": ("نوت", "note"),
    "NICE PUPPET": ("نایس پاپت", "nice puppet"),
    "Citray": ("سیترای", "citray"),
    "Super Kay": ("سوپر کی", "super kay"),
    "Bright Max": ("برایت مکس", "bright max"),
    "Defacto": ("دیفکتو", "defacto"),
    "prime": ("پرایم", "prime"),
    "Signature": ("سیگنیچر", "signature"),
    "Woody Sence": ("وودی سنس", "woody sence", "woody sense"),
    "Seniorita": ("سنیوریتا", "seniorita"),
    "Dafi": ("دافی", "dafi"),
    "RODIER": ("رودیر", "rodier"),
    "Rang Ta Rang": ("رنگ تا رنگ", "rang ta rang"),
    "Merida": ("مریدا", "merida"),
    "Noyer": ("نویر", "noyer"),
    "Vitalayer": ("ویتالیر", "vitalayer"),
    "Sclaree": ("اسکلاره", "sclaree"),
    "Cinere": ("سینره", "cinere"),
    "JUXI": ("ژاکسی", "juxi"),
    "Heritage": ("هریتیج", "heritage"),
    "Hello Fresh": ("هلو فرش", "hello fresh"),
    "Troya": ("ترویا", "تریا", "troya"),
}


def _alias_key(value: str) -> str:
    return (
        str(value or "")
        .replace("\u200c", "")
        .strip()
        .casefold()
    )


def build_brand_alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in CANONICAL_BRAND_ALIASES.items():
        mapping[_alias_key(canonical)] = canonical
        for alias in aliases:
            mapping[_alias_key(alias)] = canonical
    return mapping


BRAND_ALIAS_MAP = build_brand_alias_map()


def canonicalize_brand(value: str | None) -> str | None:
    if not value or not str(value).strip():
        return None
    return BRAND_ALIAS_MAP.get(_alias_key(value))


def brand_match_keys(canonical: str) -> list[str]:
    keys = {_alias_key(canonical)}
    for alias, mapped in BRAND_ALIAS_MAP.items():
        if mapped.casefold() == canonical.casefold():
            keys.add(alias)
            keys.add(_alias_key(mapped))
    return sorted(keys)


def brand_aliases_longest_first() -> list[tuple[str, str]]:
    return sorted(
        BRAND_ALIAS_MAP.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
