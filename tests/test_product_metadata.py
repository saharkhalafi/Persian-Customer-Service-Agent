from app.repositories.product_repository import build_product_search_sql
from app.services.product_brand_aliases import canonicalize_brand
from app.services.product_metadata import (
    ProductSearchMetadata,
    extract_product_metadata,
    search_tokens_for_metadata,
)
from app.services.product_query_normalizer import catalog_search_tokens


def test_salute_persian_request_extracts_brand_only():
    meta = extract_product_metadata("محصولات سالوته رو میخوام")
    assert meta.brand == "Salute"
    assert meta.category is None
    assert meta.price_max is None


def test_salute_english_products_extracts_brand():
    meta = extract_product_metadata("Salute products")
    assert meta.brand == "Salute"


def test_salute_uppercase_extracts_brand():
    meta = extract_product_metadata("SALUTE")
    assert meta.brand == "Salute"


def test_salute_alone_extracts_brand():
    meta = extract_product_metadata("سالوته")
    assert meta.brand == "Salute"


def test_xiaomi_persian_request_extracts_brand():
    meta = extract_product_metadata("محصولات شیائومی")
    assert meta.brand == "Xiaomi"


def test_note_under_300k_extracts_brand_and_price():
    meta = extract_product_metadata("محصولات نوت زیر 300هزارتومان")
    assert canonicalize_brand(meta.brand) == "Note"
    assert meta.brand.casefold() == "note"
    assert meta.price_max == 300000
    assert meta.price_min is None
    assert search_tokens_for_metadata("محصولات نوت زیر 300هزارتومان", meta) == []


def test_persian_and_english_brand_share_canonical():
    assert canonicalize_brand("سالوته") == canonicalize_brand("Salute")
    assert canonicalize_brand("سالوت") == "Salute"
    assert canonicalize_brand("salute") == "Salute"
    assert canonicalize_brand("SALUTE") == "Salute"
    assert canonicalize_brand("شیائومی") == canonicalize_brand("xiaomi")
    assert canonicalize_brand("Xiaomi") == "Xiaomi"


def test_uncertain_metadata_is_not_guessed():
    meta = extract_product_metadata("یه چیز خوب برای مو میخوام")
    assert meta.brand is None
    assert meta.category is None
    assert meta.price_max is None
    assert meta.color is None


def test_unknown_brand_does_not_set_filter():
    meta = extract_product_metadata("محصولات برند ناشناخته رو میخوام")
    assert meta.brand is None


def test_query_without_metadata_keeps_previous_tokens():
    query = "ماسک مو مناسب موهای خشک"
    meta = extract_product_metadata(query)
    tokens = search_tokens_for_metadata(query, meta)
    # category ماسک مو is a confident catalog type
    leftover = catalog_search_tokens(query)
    if meta.category:
        assert "ماسک" not in tokens or meta.category
    else:
        assert tokens == leftover


def test_no_metadata_token_search_unchanged_for_plain_catalog_query():
    query = "Signature"
    meta = extract_product_metadata(query)
    tokens = search_tokens_for_metadata(query, meta)
    if meta.brand == "Signature":
        assert tokens == []
    else:
        assert tokens == ["Signature"]


def test_sql_applies_brand_filter_before_limit():
    sql, params = build_product_search_sql(
        tokens=[],
        filters=ProductSearchMetadata(brand="Salute"),
        limit=10,
    )
    where_index = sql.upper().index("WHERE")
    limit_index = sql.upper().index("LIMIT")
    order_index = sql.upper().index("ORDER BY")
    assert where_index < order_index < limit_index
    assert "LOWER(TRIM(brand)) IN" in sql
    assert "DESC" in sql.upper()
    assert any(value == "salute" for value in params.values())
    assert params["limit"] == 10
    assert "term_0" not in params


def test_embedded_hair_mask_does_not_hard_filter_hair_dye():
    meta = extract_product_metadata("ماسک تثبیت کننده رنگ مو میخوام")
    assert meta.category is None
    tokens = search_tokens_for_metadata("ماسک تثبیت کننده رنگ مو میخوام", meta)
    assert "ماسک" in tokens
    assert "رنگ" in tokens


def test_embedded_hand_and_nail_cream_does_not_hard_filter_hand_cream():
    meta = extract_product_metadata("کرم دست و ناخن حاوی ویتامین سی میخوام")
    assert meta.category is None


def test_explicit_hair_dye_and_hand_cream_stay_hard_filters():
    assert extract_product_metadata("رنگ مو").category == "رنگ مو"
    assert extract_product_metadata("کرم دست").category == "کرم دست"
    assert extract_product_metadata("لاک ناخن مریدا").category == "لاک ناخن"


def test_negative_brand_query_does_not_set_positive_brand():
    meta = extract_product_metadata("لاک ناخن میخوام ولی از مریدا نه")
    assert meta.brand is None
    assert meta.category == "لاک ناخن"
    assert meta.exclude_brands == ("Merida",)


def test_sql_without_metadata_keeps_token_search():
    sql, params = build_product_search_sql(
        tokens=["ماسک", "مو"],
        filters=ProductSearchMetadata(),
        limit=10,
    )
    assert "ILIKE :term_0" in sql
    assert "brand_key_0" not in params
    assert params["term_0"] == "%ماسک%"
