from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.observability import traced

from app.services.product_brand_aliases import brand_match_keys
from app.services.product_metadata import ProductSearchMetadata
from app.services.product_name_matching import (
    name_phrase_variants,
    name_token_variants,
)
from app.services.product_query_normalizer import hard_catalog_tokens


_SEARCHABLE_TEXT = """
    concat_ws(
        ' ',
        product_name,
        brand,
        type,
        sku,
        product_code,
        category_level1,
        category_level2,
        color
    )
"""

_PRICE_NUMERIC = """
    NULLIF(
        regexp_replace(COALESCE(sum_of_price, ''), '[^0-9.]+', '', 'g'),
        ''
    )::numeric
"""

_NORMALIZED_NAME_SQL = """
    regexp_replace(
        translate(
            lower(
                trim(
                    regexp_replace(
                        COALESCE(product_name, ''),
                        E'[\\u200c[:punct:]]+',
                        ' ',
                        'g'
                    )
                )
            ),
            'يىك',
            'ییک'
        ),
        '\\s+',
        ' ',
        'g'
    )
"""


def _escape_like(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _token_name_match_sql(variants: list[str], prefix: str, params: dict[str, object]) -> str:
    clauses: list[str] = []
    for index, variant in enumerate(variants):
        like_key = f"{prefix}_like_{index}"
        raw_key = f"{prefix}_raw_{index}"
        params[like_key] = f"%{_escape_like(variant)}%"
        params[raw_key] = variant
        clauses.append(
            f"("
            f"{_NORMALIZED_NAME_SQL} LIKE :{like_key} ESCAPE '\\' "
            f"OR COALESCE(product_name, '') ILIKE :{like_key} ESCAPE '\\'"
            f")"
        )
    if not clauses:
        return "FALSE"
    return "(" + " OR ".join(clauses) + ")"


def _relevance_score_sql(
    ranking_tokens: list[str],
    filters: ProductSearchMetadata | None,
    params: dict[str, object],
    name_matching: bool = False,
    name_tokens: list[str] | None = None,
    name_phrase: str | None = None,
) -> str:
    score_parts = [
        "("
        "CASE WHEN LENGTH(TRIM(:rank_query)) > 0 "
        "AND LOWER(TRIM(COALESCE(product_name, ''))) = LOWER(TRIM(:rank_query)) "
        "THEN 100 ELSE 0 END"
        ")"
    ]

    if filters and filters.brand:
        brand_keys = [
            key
            for key, value in params.items()
            if str(key).startswith("brand_key_")
        ]
        if brand_keys:
            in_list = ", ".join(f":{key}" for key in brand_keys)
            score_parts.append(
                f"(CASE WHEN LOWER(TRIM(brand)) IN ({in_list}) THEN 50 ELSE 0 END)"
            )

    if filters and filters.category:
        score_parts.append(
            "(CASE WHEN LOWER(TRIM(type)) = LOWER(:category) THEN 40 ELSE 0 END)"
        )

    phrase = (name_phrase or params.get("rank_query") or "").strip()
    if phrase:
        params["class_phrase"] = f"%{_escape_like(phrase)}%"
        params["class_phrase_start"] = f"{_escape_like(phrase)}%"
        score_parts.append(
            "(CASE WHEN COALESCE(product_name, '') ILIKE :class_phrase_start "
            "ESCAPE '\\' THEN 18 ELSE 0 END)"
        )
        score_parts.append(
            "(CASE WHEN COALESCE(product_name, '') ILIKE :class_phrase "
            "ESCAPE '\\' THEN 12 ELSE 0 END)"
        )

    variant_tokens = [
        token
        for token in (ranking_tokens or [])
        if token and any(char.isdigit() or char in "۰۱۲۳۴۵۶۷۸۹" for char in token)
    ]
    for index, token in enumerate(variant_tokens[:4]):
        key = f"variant_term_{index}"
        params[key] = f"%{_escape_like(token)}%"
        score_parts.append(
            f"(CASE WHEN COALESCE(product_name, '') ILIKE :{key} "
            f"ESCAPE '\\' OR COALESCE(sku, '') ILIKE :{key} ESCAPE '\\' "
            "THEN 25 ELSE 0 END)"
        )

    if ranking_tokens:
        all_searchable = " AND ".join(
            f"{_SEARCHABLE_TEXT} ILIKE :rank_term_{index} ESCAPE '\\'"
            for index in range(len(ranking_tokens))
        )
        all_name = " AND ".join(
            f"COALESCE(product_name, '') ILIKE :rank_term_{index} ESCAPE '\\'"
            for index in range(len(ranking_tokens))
        )
        score_parts.append(f"(CASE WHEN {all_searchable} THEN 30 ELSE 0 END)")
        score_parts.append(f"(CASE WHEN {all_name} THEN 20 ELSE 0 END)")

        for index, token in enumerate(ranking_tokens):
            params[f"rank_term_{index}"] = f"%{_escape_like(token)}%"
            params[f"rank_raw_{index}"] = token
            score_parts.append(
                f"(CASE WHEN COALESCE(product_name, '') ILIKE :rank_term_{index} "
                "ESCAPE '\\' THEN 6 ELSE 0 END)"
            )
            score_parts.append(
                f"(CASE WHEN LOWER(TRIM(COALESCE(brand, ''))) = LOWER(:rank_raw_{index}) "
                f"OR COALESCE(brand, '') ILIKE :rank_term_{index} ESCAPE '\\' "
                "THEN 5 ELSE 0 END)"
            )
            score_parts.append(
                f"(CASE WHEN LOWER(TRIM(COALESCE(type, ''))) = LOWER(:rank_raw_{index}) "
                f"OR COALESCE(type, '') ILIKE :rank_term_{index} ESCAPE '\\' "
                f"OR COALESCE(category_level1, '') ILIKE :rank_term_{index} ESCAPE '\\' "
                f"OR COALESCE(category_level2, '') ILIKE :rank_term_{index} ESCAPE '\\' "
                "THEN 4 ELSE 0 END)"
            )
            score_parts.append(
                f"(CASE WHEN {_SEARCHABLE_TEXT} ILIKE :rank_term_{index} "
                "ESCAPE '\\' THEN 1 ELSE 0 END)"
            )

    if name_matching:
        score_parts.extend(
            _name_match_score_sql(name_tokens or [], name_phrase or "", params)
        )

    return " + ".join(score_parts)


def _name_match_score_sql(
    name_tokens: list[str],
    name_phrase: str,
    params: dict[str, object],
) -> list[str]:
    score_parts: list[str] = []
    phrases = name_phrase_variants(name_phrase)
    if phrases:
        phrase_clauses: list[str] = []
        start_clauses: list[str] = []
        for index, phrase in enumerate(phrases):
            like_key = f"name_phrase_{index}"
            start_key = f"name_phrase_start_{index}"
            params[like_key] = f"%{_escape_like(phrase)}%"
            params[start_key] = f"{_escape_like(phrase)}%"
            phrase_clauses.append(
                f"{_NORMALIZED_NAME_SQL} LIKE :{like_key} ESCAPE '\\'"
            )
            start_clauses.append(
                f"{_NORMALIZED_NAME_SQL} LIKE :{start_key} ESCAPE '\\'"
            )
        phrase_sql = " OR ".join(phrase_clauses)
        start_sql = " OR ".join(start_clauses)
        score_parts.append(f"(CASE WHEN {phrase_sql} THEN 80 ELSE 0 END)")
        score_parts.append(f"(CASE WHEN {start_sql} THEN 35 ELSE 0 END)")
        params["name_phrase_len"] = len(phrases[0])
        score_parts.append(
            "("
            f"CASE WHEN {phrase_sql} THEN LEAST("
            "40, "
            "ROUND("
            "(CAST(:name_phrase_len AS numeric) / "
            "GREATEST(char_length(TRIM(COALESCE(product_name, ''))), 1)"
            ") * 55, 2)"
            ") ELSE 0 END"
            ")"
        )
        score_parts.append(
            "("
            "CASE WHEN LENGTH(TRIM(:rank_query)) > 0 "
            f"AND {_NORMALIZED_NAME_SQL} = LOWER(TRIM(:rank_query)) "
            "THEN 20 ELSE 0 END"
            ")"
        )

    token_match_sqls: list[str] = []
    for index, token in enumerate(name_tokens):
        variants = name_token_variants(token)
        match_sql = _token_name_match_sql(variants, f"name_tok_{index}", params)
        token_match_sqls.append(match_sql)
        score_parts.append(f"(CASE WHEN {match_sql} THEN 8 ELSE 0 END)")
        if len(token) >= 4:
            prefix_key = f"name_tok_{index}_prefix"
            params[prefix_key] = f"%{_escape_like(token[:-1])}%"
            score_parts.append(
                f"(CASE WHEN NOT {match_sql} AND {_NORMALIZED_NAME_SQL} "
                f"LIKE :{prefix_key} ESCAPE '\\' THEN 4 ELSE 0 END)"
            )

    if token_match_sqls:
        all_in_name = " AND ".join(token_match_sqls)
        score_parts.append(f"(CASE WHEN {all_in_name} THEN 50 ELSE 0 END)")
        coverage = " + ".join(
            f"(CASE WHEN {clause} THEN 1 ELSE 0 END)"
            for clause in token_match_sqls
        )
        params["name_token_count"] = len(token_match_sqls)
        score_parts.append(
            f"(ROUND((({coverage})::numeric / CAST(:name_token_count AS numeric)) * 40, 2))"
        )

    return score_parts


def _name_match_score_only_sql(
    name_tokens: list[str],
    name_phrase: str,
    params: dict[str, object],
) -> str:
    parts = _name_match_score_sql(name_tokens, name_phrase, params)
    return " + ".join(parts) if parts else "0"


def _hard_filter_conditions(
    tokens: list[str],
    filters: ProductSearchMetadata | None,
    params: dict[str, object],
    apply_token_filters: bool,
) -> list[str]:
    conditions: list[str] = []

    if filters and filters.brand:
        brand_keys = brand_match_keys(filters.brand)
        placeholders = []
        for index, key in enumerate(brand_keys):
            name = f"brand_key_{index}"
            placeholders.append(f":{name}")
            params[name] = key
        conditions.append(
            f"LOWER(TRIM(brand)) IN ({', '.join(placeholders)})"
        )

    if filters and filters.exclude_brands:
        excluded_keys: list[str] = []
        for brand in filters.exclude_brands:
            excluded_keys.extend(brand_match_keys(brand))
        excluded_keys = sorted(set(excluded_keys))
        placeholders = []
        for index, key in enumerate(excluded_keys):
            name = f"exclude_brand_{index}"
            placeholders.append(f":{name}")
            params[name] = key
        if placeholders:
            conditions.append(
                f"(brand IS NULL OR LOWER(TRIM(brand)) NOT IN ({', '.join(placeholders)}))"
            )

    if filters and filters.category:
        conditions.append(
            "("
            "LOWER(TRIM(type)) = LOWER(:category) "
            "OR type ILIKE :category_like ESCAPE '\\' "
            "OR category_level1 ILIKE :category_like ESCAPE '\\' "
            "OR category_level2 ILIKE :category_like ESCAPE '\\'"
            ")"
        )
        params["category"] = filters.category
        params["category_like"] = f"%{_escape_like(filters.category)}%"

    if filters and filters.color:
        conditions.append("color ILIKE :color ESCAPE '\\'")
        params["color"] = f"%{_escape_like(filters.color)}%"

    if filters and filters.material:
        conditions.append(
            f"{_SEARCHABLE_TEXT} ILIKE :material ESCAPE '\\'"
        )
        params["material"] = f"%{_escape_like(filters.material)}%"

    if filters and filters.gender:
        conditions.append("gender ILIKE :gender ESCAPE '\\'")
        params["gender"] = f"%{_escape_like(filters.gender)}%"

    if filters and filters.price_min is not None:
        conditions.append(f"{_PRICE_NUMERIC} >= :price_min")
        params["price_min"] = filters.price_min

    if filters and filters.price_max is not None:
        conditions.append(f"{_PRICE_NUMERIC} <= :price_max")
        params["price_max"] = filters.price_max

    if filters and filters.availability is True:
        conditions.append("last_status = 'Enable'")
    elif filters and filters.availability is False:
        conditions.append("last_status IS DISTINCT FROM 'Enable'")

    if apply_token_filters:
        for index, term in enumerate(tokens):
            key = f"term_{index}"
            conditions.append(
                f"{_SEARCHABLE_TEXT} ILIKE :{key} ESCAPE '\\'"
            )
            params[key] = f"%{_escape_like(term)}%"

    return conditions


@dataclass
class ProductSearchQuery:
    terms: list[str]
    apply_token_filters: bool
    conditions: list[str]
    params: dict[str, object]
    relevance_sql: str
    name_score_sql: str
    restrict_name_pool: bool
    empty: bool


def resolve_product_search_query(
    tokens: list[str] | None = None,
    filters: ProductSearchMetadata | None = None,
    query: str | None = None,
    ranking_tokens: list[str] | None = None,
    rank_query: str | None = None,
    name_matching: bool = True,
    name_tokens: list[str] | None = None,
    name_phrase: str | None = None,
    limit: int = 10,
    candidate_limit: int = 80,
) -> ProductSearchQuery:
    terms = [token.strip() for token in (tokens or []) if token and token.strip()]
    if not terms and query and not (filters and filters.has_filters()):
        terms = [query.strip()]
    terms = hard_catalog_tokens(terms)[:12]
    rank_tokens = [
        token.strip()
        for token in (ranking_tokens or [])
        if token and token.strip()
    ][:12]
    resolved_name_tokens = [
        token.strip()
        for token in (name_tokens or rank_tokens)
        if token and token.strip()
    ][:12]
    resolved_name_phrase = (name_phrase or rank_query or " ".join(resolved_name_tokens)).strip()
    params: dict[str, object] = {
        "limit": limit,
        "candidate_limit": candidate_limit,
    }
    apply_token_filters = bool(terms) and not (
        filters and filters.has_positive_filters()
    )
    conditions = _hard_filter_conditions(
        terms,
        filters,
        params,
        apply_token_filters=apply_token_filters,
    )
    empty = False
    if (
        filters
        and filters.exclude_brands
        and not filters.has_positive_filters()
        and not terms
        and not (query or "").strip()
    ):
        empty = True
    if not conditions:
        empty = True
    params["rank_query"] = (rank_query or resolved_name_phrase or " ".join(rank_tokens)).strip()
    relevance_sql = "0"
    name_score_sql = "0"
    if not empty:
        relevance_sql = _relevance_score_sql(
            rank_tokens,
            filters,
            params,
            name_matching=name_matching,
            name_tokens=resolved_name_tokens,
            name_phrase=resolved_name_phrase,
        )
        name_score_sql = (
            _name_match_score_only_sql(resolved_name_tokens, resolved_name_phrase, params)
            if name_matching
            else "0"
        )
    restrict_name_pool = bool(
        name_matching
        and resolved_name_tokens
        and filters
        and filters.has_positive_filters()
        and terms
    )
    return ProductSearchQuery(
        terms=terms,
        apply_token_filters=apply_token_filters,
        conditions=conditions,
        params=params,
        relevance_sql=relevance_sql,
        name_score_sql=name_score_sql,
        restrict_name_pool=restrict_name_pool,
        empty=empty,
    )


_SELECT_COLUMNS = """
                product_code,
                product_name,
                brand,
                type,
                sku,
                category_level1,
                category_level2,
                last_status,
                available_qty,
                special_price,
                sum_of_price,
                color,
                size,
                gender
"""


def build_product_search_sql(
    tokens: list[str] | None = None,
    filters: ProductSearchMetadata | None = None,
    limit: int = 10,
    query: str | None = None,
    ranking_tokens: list[str] | None = None,
    rank_query: str | None = None,
    name_matching: bool = True,
    name_tokens: list[str] | None = None,
    name_phrase: str | None = None,
    candidate_limit: int = 80,
    max_fetch: int = 50,
) -> tuple[str, dict[str, object]]:
    limit = min(max(limit, 1), max_fetch)
    candidate_limit = min(max(candidate_limit, limit), 200)
    resolved = resolve_product_search_query(
        tokens=tokens,
        filters=filters,
        query=query,
        ranking_tokens=ranking_tokens,
        rank_query=rank_query,
        name_matching=name_matching,
        name_tokens=name_tokens,
        name_phrase=name_phrase,
        limit=limit,
        candidate_limit=candidate_limit,
    )
    if resolved.empty:
        return "", resolved.params
    where_sql = " AND ".join(resolved.conditions)
    order_sql = f"""
                ({resolved.relevance_sql}) DESC,
                CASE
                    WHEN last_status = 'Enable' THEN 1
                    ELSE 0
                END DESC,
                product_name ASC NULLS LAST
    """
    if resolved.restrict_name_pool:
        sql = f"""
            WITH filtered AS (
                SELECT
                    {_SELECT_COLUMNS},
                    ({resolved.name_score_sql}) AS name_match_score,
                    ({resolved.relevance_sql}) AS relevance_score
                FROM public.products
                WHERE
                    {where_sql}
            ),
            candidates AS (
                SELECT *
                FROM filtered
                ORDER BY
                    name_match_score DESC,
                    CASE
                        WHEN last_status = 'Enable' THEN 1
                        ELSE 0
                    END DESC,
                    product_name ASC NULLS LAST
                LIMIT :candidate_limit
            )
            SELECT
                {_SELECT_COLUMNS}
            FROM candidates
            ORDER BY
                relevance_score DESC,
                CASE
                    WHEN last_status = 'Enable' THEN 1
                    ELSE 0
                END DESC,
                product_name ASC NULLS LAST
            LIMIT :limit
        """
    else:
        sql = f"""
            SELECT
                {_SELECT_COLUMNS}
            FROM public.products
            WHERE
                {where_sql}
            ORDER BY
                {order_sql}
            LIMIT :limit
        """
    return sql, resolved.params


class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    @traced("db_operation", layer="product_repository", operation="search_products")
    def search_products(
        self,
        query: str | None = None,
        limit: int = 10,
        tokens: list[str] | None = None,
        filters: ProductSearchMetadata | None = None,
        ranking_tokens: list[str] | None = None,
        rank_query: str | None = None,
        name_matching: bool = True,
        name_tokens: list[str] | None = None,
        name_phrase: str | None = None,
        candidate_limit: int = 80,
        max_fetch: int = 50,
    ) -> list[dict]:
        sql, params = build_product_search_sql(
            tokens=tokens,
            filters=filters,
            limit=limit,
            query=query,
            ranking_tokens=ranking_tokens,
            rank_query=rank_query,
            name_matching=name_matching,
            name_tokens=name_tokens,
            name_phrase=name_phrase,
            candidate_limit=candidate_limit,
            max_fetch=max_fetch,
        )
        if not sql:
            return []

        result = self.db.execute(text(sql), params).mappings()
        return [dict(row) for row in result]

    def inspect_filtered_candidates(
        self,
        query: str | None = None,
        tokens: list[str] | None = None,
        filters: ProductSearchMetadata | None = None,
        ranking_tokens: list[str] | None = None,
        rank_query: str | None = None,
        name_matching: bool = False,
        name_tokens: list[str] | None = None,
        name_phrase: str | None = None,
        gold_codes: list[str] | None = None,
    ) -> dict:
        resolved = resolve_product_search_query(
            tokens=tokens,
            filters=filters,
            query=query,
            ranking_tokens=ranking_tokens,
            rank_query=rank_query,
            name_matching=name_matching,
            name_tokens=name_tokens,
            name_phrase=name_phrase,
            limit=1,
            candidate_limit=80,
        )
        if resolved.empty:
            return {
                "hard_filtered_count": 0,
                "apply_token_filters": resolved.apply_token_filters,
                "token_filters": resolved.terms if resolved.apply_token_filters else [],
                "where_sql": "",
                "gold_in_filtered": [],
                "restrict_name_pool": False,
            }
        where_sql = " AND ".join(resolved.conditions)
        count = self.db.execute(
            text(f"SELECT COUNT(*) FROM public.products WHERE {where_sql}"),
            resolved.params,
        ).scalar()
        gold_in_filtered: list[str] = []
        codes = [str(code) for code in (gold_codes or []) if str(code).strip()]
        if codes:
            found = self.db.execute(
                text(
                    f"""
                    SELECT product_code
                    FROM public.products
                    WHERE {where_sql}
                      AND product_code = ANY(:gold_codes)
                    """
                ),
                {**resolved.params, "gold_codes": codes},
            ).scalars()
            gold_in_filtered = [str(code) for code in found]
        return {
            "hard_filtered_count": int(count or 0),
            "apply_token_filters": resolved.apply_token_filters,
            "token_filters": resolved.terms if resolved.apply_token_filters else [],
            "where_sql": where_sql,
            "gold_in_filtered": gold_in_filtered,
            "restrict_name_pool": resolved.restrict_name_pool,
        }

    def fetch_products_by_codes(self, codes: list[str]) -> dict[str, dict]:
        clean = [str(code) for code in codes if str(code).strip()]
        if not clean:
            return {}
        rows = self.db.execute(
            text(
                """
                SELECT
                    product_code, product_name, brand, type, sku,
                    category_level1, category_level2, last_status,
                    available_qty, special_price, sum_of_price,
                    color, size, gender
                FROM public.products
                WHERE product_code = ANY(:codes)
                """
            ),
            {"codes": clean},
        ).mappings()
        return {str(row["product_code"]): dict(row) for row in rows}
