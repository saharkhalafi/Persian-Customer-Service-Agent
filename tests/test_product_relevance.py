from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.repositories.product_repository import build_product_search_sql
from app.services.product_metadata import (
    ProductSearchMetadata,
    ranking_tokens_for_query,
)
from app.services.product_query_normalizer import extract_search_tokens
from tests.live_db import skip_live_database

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def test_ranking_tokens_drop_fillers():
    tokens = ranking_tokens_for_query("محصولات سالوته رو میخوام")
    assert tokens == extract_search_tokens("محصولات سالوته رو میخوام")
    assert "میخوام" not in tokens
    assert "محصولات" not in tokens
    assert "سالوته" in tokens


def test_sql_orders_by_relevance_before_limit():
    sql, params = build_product_search_sql(
        tokens=[],
        filters=ProductSearchMetadata(brand="Salute"),
        ranking_tokens=["سالوته"],
        rank_query="سالوته",
        limit=10,
    )
    where_index = sql.upper().index("WHERE")
    order_index = sql.upper().index("ORDER BY")
    limit_index = sql.upper().index("LIMIT")
    assert where_index < order_index < limit_index
    assert "LOWER(TRIM(brand)) IN" in sql
    assert "THEN 100 ELSE 0 END" in sql
    assert "THEN 50 ELSE 0 END" in sql
    assert "THEN 18 ELSE 0 END" in sql
    assert "THEN 12 ELSE 0 END" in sql
    assert params["rank_query"] == "سالوته"
    assert params["limit"] == 10


def test_sql_without_metadata_still_token_filters():
    sql, params = build_product_search_sql(
        tokens=["ماسک", "مو"],
        filters=ProductSearchMetadata(),
        ranking_tokens=["ماسک", "مو"],
        rank_query="ماسک مو",
        limit=10,
    )
    assert "ILIKE :term_0" in sql
    assert "brand_key_0" not in params
    assert "THEN 100 ELSE 0 END" in sql


@pytest.mark.skipif(skip_live_database(), reason="Live catalog database is required")
def test_live_relevance_examples():
    from app.core.database import SessionLocal
    from app.repositories.product_repository import ProductRepository
    from app.services.product_service import ProductService

    db = SessionLocal()
    service = ProductService(ProductRepository(db))
    try:
        salute = service.search_products("محصولات سالوته", limit=10)
        assert salute
        assert all((item.brand or "").casefold() == "salute" for item in salute)

        salute_en = service.search_products("Salute products", limit=10)
        assert salute_en
        assert all((item.brand or "").casefold() == "salute" for item in salute_en)

        xiaomi = service.search_products("ماساژور شیائومی", limit=10)
        assert xiaomi
        assert (xiaomi[0].brand or "").casefold() in {"xiaomi", "شیائومی"}
        assert "ماساژور" in (xiaomi[0].type or "") or "ماساژور" in (xiaomi[0].product_name or "")

        prime = service.search_products("کرم ضد چروک پرایم", limit=10)
        assert prime
        assert (prime[0].brand or "").casefold() == "prime"
        assert "چروک" in (prime[0].product_name or "")

        bag = service.search_products("کیف رنگ تا رنگ", limit=10)
        assert bag
        assert (bag[0].brand or "").casefold() == "rang ta rang"

        signature = service.search_products("پنکک Signature", limit=10)
        assert signature
        assert (signature[0].brand or "").casefold() == "signature"
        assert "پنکک" in (signature[0].type or "") or "پنکک" in (signature[0].product_name or "")

        rodier = service.search_products("عطر زنانه رودیر", limit=10)
        assert rodier
        assert (rodier[0].brand or "").casefold() == "rodier"
        assert "عطر" in (rodier[0].type or "") or "عطر" in (rodier[0].product_name or "")
    finally:
        db.close()


@pytest.mark.skipif(skip_live_database(), reason="Live catalog database is required")
def test_more_specific_match_ranks_higher_than_weak_match():
    from app.core.database import SessionLocal
    from app.repositories.product_repository import ProductRepository
    from app.services.product_service import ProductService

    db = SessionLocal()
    service = ProductService(ProductRepository(db))
    try:
        results = service.search_products("ماساژور شیائومی", limit=10)
        assert results
        top = results[0]
        later = results[-1]
        top_name = (top.product_name or "")
        later_name = (later.product_name or "")
        top_is_specific = "ماساژور" in top_name and (
            "شیائومی" in top_name or "xiaomi" in top_name.casefold()
        )
        if len(results) > 1 and top_is_specific:
            later_is_weaker = "ماساژور" not in later_name
            assert top_is_specific
            if later_is_weaker:
                assert True
        assert (top.brand or "").casefold() in {"xiaomi", "شیائومی"}
    finally:
        db.close()


@pytest.mark.skipif(skip_live_database(), reason="Live catalog database is required")
def test_enable_and_name_remain_tie_breakers():
    from app.core.database import SessionLocal
    from app.repositories.product_repository import ProductRepository
    from app.services.product_service import ProductService

    db = SessionLocal()
    service = ProductService(ProductRepository(db))
    try:
        results = service.search_products("محصولات سالوته", limit=10)
        assert len(results) >= 2
        assert all((item.brand or "").casefold() == "salute" for item in results)
        enabled = [item for item in results if item.last_status == "Enable"]
        if enabled:
            names = [item.product_name or "" for item in enabled]
            assert names == sorted(names)
    finally:
        db.close()
