from app.services.product_query_normalizer import (
    catalog_search_tokens,
    extract_search_tokens,
    fold_search_text,
    hard_catalog_tokens,
    normalize_product_query,
)


def test_massage_xiaomi_keeps_product_terms():
    tokens = extract_search_tokens("ماساژور شیائومی میخوام")
    assert "ماساژور" in tokens
    assert "شیائومی" in tokens
    assert "میخوام" not in tokens


def test_salute_generic_request_keeps_brand():
    assert normalize_product_query("محصولات سالوته رو میخوام") == "سالوته"


def test_please_show_sports_shoes():
    tokens = extract_search_tokens("لطفاً کفش ورزشی بهم نشون بده")
    assert "کفش" in tokens
    assert "ورزشی" in tokens


def test_price_constraint_is_preserved():
    tokens = extract_search_tokens("کفش ورزشی نایک زیر ۲ میلیون میخوام")
    assert "کفش" in tokens
    assert "ورزشی" in tokens
    assert "نایک" in tokens
    assert "۲" in tokens
    assert "میلیون" in tokens


def test_signature_stays_searchable():
    assert normalize_product_query("Signature") == "Signature"


def test_salute_alone_unchanged():
    assert normalize_product_query("سالوته") == "سالوته"


def test_nice_puppet_keeps_both_tokens():
    tokens = extract_search_tokens("نایس پاپت")
    assert tokens == ["نایس", "پاپت"]


def test_question_mark_is_not_a_token():
    tokens = extract_search_tokens("از برند نایس پاپت چی دارید؟")
    assert tokens == ["نایس", "پاپت"]
    assert "؟" not in tokens


def test_persian_number_word_is_preserved():
    tokens = extract_search_tokens("زیر یک میلیون تومن میخوام")
    assert "یک" in tokens
    assert "میلیون" in tokens
    assert "تومن" in tokens


def test_arabic_letters_fold_to_persian():
    assert fold_search_text("يك") == "یک"
    assert "ک" in fold_search_text("كرم")


def test_soft_tokens_stay_in_ranking_but_not_hard_filters():
    tokens = extract_search_tokens("برای مرطوب کردن دست چی دارید؟")
    assert "مرطوب" in tokens
    assert "دست" in tokens
    assert "کردن" in tokens
    assert "کردن" not in hard_catalog_tokens(tokens)
    assert "مرطوب" in hard_catalog_tokens(tokens)


def test_catalog_tokens_keep_product_terms_not_price_words():
    catalog = catalog_search_tokens("کفش ورزشی نایک زیر ۲ میلیون میخوام")
    assert "کفش" in catalog
    assert "ورزشی" in catalog
    assert "نایک" in catalog
    assert "۲" not in catalog
    assert "میلیون" not in catalog
