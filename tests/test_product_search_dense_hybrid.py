from app.services.product_metadata import ProductSearchMetadata
from evaluation.product_search_dense_hybrid import (
    bm25_scores,
    build_bm25_stats,
    min_max_normalize,
    row_matches_metadata,
    searchable_text,
    tokenize,
)
import numpy as np


def test_hard_brand_filter_keeps_matching_row():
    row = {
        "brand": "Defacto",
        "type": "کرم دست",
        "category_level1": "",
        "category_level2": "",
        "color": "",
        "gender": "",
        "sum_of_price": "100000",
        "last_status": "Enable",
        "_search": "کرم دست Defacto",
        "_search_fold": searchable_text(
            {"product_name": "کرم دست", "brand": "Defacto", "type": "کرم دست"}
        ).casefold(),
    }
    row["_search_fold"] = row["_search"].casefold()
    assert row_matches_metadata(
        row,
        ProductSearchMetadata(brand="Defacto", category="کرم دست"),
        [],
    )
    assert not row_matches_metadata(
        row,
        ProductSearchMetadata(brand="Salute", category="کرم دست"),
        [],
    )


def test_minmax_and_bm25_are_deterministic():
    values = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    norm = min_max_normalize(values)
    assert abs(float(norm[0]) - 0.0) < 1e-6
    assert abs(float(norm[-1]) - 1.0) < 1e-6
    docs = [tokenize("لاک ناخن مریدا"), tokenize("کرم دست سالوته"), tokenize("لاک ناخن")]
    idf, avgdl = build_bm25_stats(docs)
    scores = bm25_scores(["لاک", "ناخن"], docs, [0, 1, 2], idf, avgdl)
    assert scores[0] > scores[1]
