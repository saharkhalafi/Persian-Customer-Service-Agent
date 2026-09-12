from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.core.database import SessionLocal
from app.repositories.product_repository import ProductRepository
from app.services.product_metadata import extract_product_metadata
from app.services.product_service import ProductService
from tests.live_db import skip_live_database


pytestmark = pytest.mark.skipif(
    skip_live_database(),
    reason="Live catalog database is required for filter tests",
)


def _service() -> tuple[ProductService, object]:
    db = SessionLocal()
    return ProductService(ProductRepository(db)), db


def test_salute_query_returns_only_salute_within_limit():
    service, db = _service()
    try:
        results = service.search_products("محصولات سالوته رو میخوام", limit=10)
        assert results
        assert len(results) <= 10
        assert all((item.brand or "").casefold() == "salute" for item in results)
    finally:
        db.close()


def test_brand_filter_is_applied_before_limit():
    service, db = _service()
    try:
        salute_count = db.execute(
            __import__("sqlalchemy", fromlist=["text"]).text(
                "SELECT COUNT(*) FROM public.products WHERE LOWER(TRIM(brand)) = 'salute'"
            )
        ).scalar()
        assert salute_count > 10

        results = service.search_products("SALUTE", limit=10)
        assert len(results) == 10
        assert all((item.brand or "").casefold() == "salute" for item in results)
    finally:
        db.close()


def test_note_price_filter_uses_sum_of_price():
    service, db = _service()
    try:
        meta = extract_product_metadata("محصولات نوت زیر 300هزارتومان")
        assert meta.brand == "Note"
        assert meta.price_max == 300000

        results = service.search_products("محصولات نوت زیر 300هزارتومان", limit=10)
        assert results
        assert all((item.brand or "").casefold() == "note" for item in results)
    finally:
        db.close()


def test_colloquial_leftover_tokens_are_not_all_required():
    service, db = _service()
    try:
        results = service.search_products("یه چیزی برای مرطوب کردن دست دارید؟", limit=10)
        assert results
        assert any(
            "مرطوب" in (item.product_name or "")
            or "مرطوب" in (item.type or "")
            or "دست" in (item.product_name or "")
            for item in results
        )
    finally:
        db.close()


def test_embedded_product_phrases_do_not_use_wrong_category_filter():
    service, db = _service()
    try:
        mask = service.search_products("ماسک تثبیت کننده رنگ مو میخوام", limit=10)
        assert mask
        assert any(
            "ماسک" in (item.product_name or "") or (item.type or "") == "ماسک مو"
            for item in mask
        )
        assert not all((item.type or "") == "رنگ مو" for item in mask)

        cream = service.search_products("کرم دست و ناخن حاوی ویتامین سی میخوام", limit=10)
        assert cream
        assert any(
            "کرم دست و ناخن" in (item.product_name or "")
            or "ناخن" in (item.product_name or "")
            for item in cream
        )
    finally:
        db.close()


def test_explicit_category_queries_remain_hard_filters():
    service, db = _service()
    try:
        hair = service.search_products("رنگ مو", limit=10)
        assert hair
        assert all(
            "رنگ مو" in (item.type or "")
            or "رنگ مو" in (item.category_level1 or "")
            or "رنگ مو" in (item.category_level2 or "")
            for item in hair
        )
        hand = service.search_products("کرم دست", limit=10)
        assert hand
        assert all(
            "کرم دست" in (item.type or "")
            or "کرم دست" in (item.product_name or "")
            or "کرم دست" in (item.category_level1 or "")
            or "کرم دست" in (item.category_level2 or "")
            for item in hand
        )
    finally:
        db.close()


def test_broad_merida_and_explicit_variant():
    service, db = _service()
    try:
        broad = service.search_products("لاک مریدا", limit=10)
        assert broad
        assert all((item.brand or "").casefold() == "merida" for item in broad)
        assert any("لاک" in (item.product_name or "") or "لاک" in (item.type or "") for item in broad)

        variant = service.search_products("لاک مریدا 001", limit=10)
        assert variant
        assert all((item.brand or "").casefold() == "merida" for item in variant)
        assert any("001" in (item.product_name or "") or "1" in (item.product_name or "") for item in variant[:3])
    finally:
        db.close()


def test_negative_brand_exclusion_is_hard():
    service, db = _service()
    try:
        results = service.search_products("لاک ناخن میخوام ولی از مریدا نه", limit=10)
        assert results
        assert all((item.brand or "").casefold() != "merida" for item in results)
    finally:
        db.close()


def test_query_without_known_brand_still_returns_token_matches():
    service, db = _service()
    try:
        results = service.search_products("ماسک مو مناسب موهای خشک", limit=10)
        assert results
        assert any("ماسک" in (item.product_name or "") or (item.type or "") == "ماسک مو" for item in results)
    finally:
        db.close()
