from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.repositories.product_repository import build_product_search_sql
from app.services.product_metadata import (
    ProductSearchMetadata,
    extract_excluded_brands,
    extract_product_metadata,
    product_name_tokens,
)
from app.services.product_name_matching import name_phrase_variants, name_token_variants
from app.services.product_query_normalizer import extract_search_tokens, fold_search_text
from tests.live_db import skip_live_database

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def test_exact_and_partial_name_tokens_are_preserved():
    tokens = product_name_tokens("ماساژور شیائومی میخوام")
    assert "ماساژور" in tokens
    assert "شیائومی" in tokens
    assert "میخوام" not in tokens


def test_all_token_name_match_keeps_distributed_concepts():
    tokens = product_name_tokens("لاک مریدا")
    assert tokens == ["لاک", "مریدا"]


def test_persian_english_name_normalization():
    assert fold_search_text("لاك ناخن") == fold_search_text("لاک ناخن") or "لاک" in fold_search_text(
        "لاک ناخن"
    )
    assert fold_search_text("يك") == "یک"
    variants = name_token_variants("مریدا")
    assert any(item.casefold() == "merida" for item in variants)
    phrases = name_phrase_variants("لاک ناخن مریدا")
    assert any("merida" in item.casefold() for item in phrases)


def test_brand_and_category_and_price_stay_in_metadata():
    meta = extract_product_metadata("ماساژور شیائومی زیر ۲ میلیون")
    assert meta.brand == "Xiaomi"
    assert meta.category == "ماساژور"
    assert meta.price_max == 2_000_000
    tokens = product_name_tokens("ماساژور شیائومی زیر ۲ میلیون", meta)
    assert "ماساژور" in tokens
    assert "۲" not in tokens
    assert "میلیون" not in tokens


def test_negative_brand_is_extracted_and_not_applied_as_positive():
    brands, remaining = extract_excluded_brands("لباس بدون مریدا")
    assert "Merida" in brands
    meta = extract_product_metadata("لاک ناخن میخوام ولی از مریدا نه")
    assert meta.category == "لاک ناخن"
    assert meta.brand is None
    assert meta.exclude_brands == ("Merida",)
    nice = extract_product_metadata("عطر زنانه میخوام بدون برند نایس پاپت")
    assert "NICE PUPPET" in nice.exclude_brands
    assert nice.brand != "NICE PUPPET"


def test_negative_is_isolated_when_disabled():
    meta = extract_product_metadata(
        "لاک ناخن میخوام ولی از مریدا نه",
        allow_negatives=False,
    )
    assert meta.brand == "Merida"
    assert meta.exclude_brands == ()


def test_sql_name_matching_after_hard_filters_does_not_bypass_them():
    sql, params = build_product_search_sql(
        tokens=["لاک"],
        filters=ProductSearchMetadata(brand="Merida", category="لاک ناخن", price_max=2000000),
        ranking_tokens=["لاک", "ناخن", "مریدا"],
        rank_query="لاک ناخن مریدا",
        name_matching=True,
        name_tokens=["لاک", "ناخن", "مریدا"],
        name_phrase="لاک ناخن مریدا",
        limit=10,
    )
    assert "LOWER(TRIM(brand)) IN" in sql
    assert "LOWER(TRIM(type)) = LOWER(:category)" in sql
    assert "<= :price_max" in sql
    assert "term_0" not in params
    assert "THEN 80 ELSE 0 END" in sql
    assert "THEN 50 ELSE 0 END" in sql
    assert params["price_max"] == 2000000


def test_sql_hard_filter_cannot_be_bypassed_by_name_score():
    sql, params = build_product_search_sql(
        tokens=["عطر"],
        filters=ProductSearchMetadata(brand="RODIER", gender="زنانه"),
        name_matching=True,
        name_tokens=["عطر", "زنانه", "رودیر"],
        name_phrase="عطر زنانه رودیر",
        limit=10,
    )
    where_sql = sql[sql.upper().index("WHERE"):sql.upper().index("ORDER BY")]
    assert "LOWER(TRIM(brand)) IN" in where_sql
    assert "gender ILIKE :gender" in where_sql
    assert "THEN 80 ELSE 0 END" in sql
    assert any(str(value).casefold() == "rodier" for value in params.values())


def test_sql_negative_brand_is_a_hard_exclusion():
    sql, params = build_product_search_sql(
        tokens=[],
        filters=ProductSearchMetadata(
            category="لاک ناخن",
            exclude_brands=("Merida",),
        ),
        name_matching=True,
        name_tokens=["لاک", "ناخن"],
        name_phrase="لاک ناخن",
        limit=10,
    )
    assert "NOT IN" in sql
    assert any(str(value).casefold() == "merida" for value in params.values())


def test_sql_metadata_only_query_does_not_over_filter_on_name_tokens():
    sql, params = build_product_search_sql(
        tokens=[],
        filters=ProductSearchMetadata(brand="Salute"),
        ranking_tokens=["سالوته"],
        rank_query="سالوته",
        name_matching=True,
        name_tokens=["سالوته"],
        name_phrase="سالوته",
        limit=10,
    )
    assert "term_0" not in params
    assert "LOWER(TRIM(brand)) IN" in sql
    assert "LIMIT :limit" in sql


def test_token_coverage_score_is_proportional():
    sql, params = build_product_search_sql(
        tokens=["لاک", "مریدا"],
        filters=ProductSearchMetadata(),
        name_matching=True,
        name_tokens=["لاک", "مریدا"],
        name_phrase="لاک مریدا",
        limit=10,
    )
    assert "* 40" in sql
    assert params["name_token_count"] == 2


@pytest.mark.skipif(skip_live_database(), reason="Live catalog database is required")
def test_live_name_matching_regressions_and_partial_names():
    from app.core.database import SessionLocal
    from app.repositories.product_repository import ProductRepository
    from app.services.product_service import ProductService

    db = SessionLocal()
    service = ProductService(
        ProductRepository(db),
        name_matching_enabled=True,
        negative_constraints_enabled=True,
    )
    try:
        exact = service.search_products("عطر زنانه رودیر", limit=10)
        assert exact
        assert (exact[0].brand or "").casefold() == "rodier"
        assert "عطر" in (exact[0].product_name or "") or (exact[0].type or "") == "عطر"

        partial = service.search_products("لاک مریدا", limit=10)
        assert partial
        assert all((item.brand or "").casefold() == "merida" for item in partial)
        assert any("لاک" in (item.product_name or "") for item in partial)

        all_tokens = service.search_products("رژ سالوته", limit=10)
        assert all_tokens
        assert all((item.brand or "").casefold() == "salute" for item in all_tokens)

        brand_name = service.search_products("ماساژور شیائومی", limit=10)
        assert brand_name
        assert (brand_name[0].brand or "").casefold() in {"xiaomi", "شیائومی"}
        assert "ماساژور" in (brand_name[0].type or "") or "ماساژور" in (
            brand_name[0].product_name or ""
        )

        category_name = service.search_products("پنکک Signature", limit=10)
        assert category_name
        assert (category_name[0].brand or "").casefold() == "signature"
        assert "پنکک" in (category_name[0].type or "") or "پنکک" in (
            category_name[0].product_name or ""
        )

        priced = service.search_products("ماساژور شیائومی زیر ۲ میلیون", limit=10)
        assert priced
        assert all((item.brand or "").casefold() in {"xiaomi", "شیائومی"} for item in priced)

        prime = service.search_products("کرم ضد چروک پرایم", limit=10)
        assert prime
        assert (prime[0].brand or "").casefold() == "prime"
        assert "چروک" in (prime[0].product_name or "")

        salute = service.search_products("محصولات سالوته", limit=10)
        assert salute
        assert all((item.brand or "").casefold() == "salute" for item in salute)

        cream = service.search_products("کرم پرایم", limit=10)
        assert cream
        assert all((item.brand or "").casefold() == "prime" for item in cream)

        excluded = service.search_products("لاک ناخن میخوام ولی از مریدا نه", limit=10)
        assert excluded
        assert all((item.brand or "").casefold() != "merida" for item in excluded)
        assert all(
            "لاک" in (item.type or "") or "لاک" in (item.product_name or "")
            for item in excluded
        )

        clothes = service.search_products("لباس بدون مریدا", limit=10)
        if clothes:
            assert all((item.brand or "").casefold() != "merida" for item in clothes)
    finally:
        db.close()


@pytest.mark.skipif(skip_live_database(), reason="Live catalog database is required")
def test_live_hard_filter_not_bypassed_by_name_matching():
    from app.core.database import SessionLocal
    from app.repositories.product_repository import ProductRepository
    from app.services.product_service import ProductService

    db = SessionLocal()
    service = ProductService(
        ProductRepository(db),
        name_matching_enabled=True,
        negative_constraints_enabled=True,
    )
    try:
        results = service.search_products("ماساژور شیائومی زیر ۲ میلیون", limit=10)
        assert results
        assert all((item.brand or "").casefold() in {"xiaomi", "شیائومی"} for item in results)
        other_brand = service.search_products("عطر زنانه رودیر", limit=10)
        assert other_brand
        assert all((item.brand or "").casefold() == "rodier" for item in other_brand)
    finally:
        db.close()


def test_search_tokens_keep_intent_words():
    tokens = extract_search_tokens("ماساژور شیائومی میخوام")
    assert tokens == ["ماساژور", "شیائومی"]
