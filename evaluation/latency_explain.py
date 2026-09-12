"""Run EXPLAIN ANALYZE for product/order queries. Does not change data."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import text

from app.core.database import SessionLocal
from app.repositories.product_repository import build_product_search_sql
from app.services.product_metadata import (
    extract_product_metadata,
    product_name_phrase,
    product_name_tokens,
    ranking_query_text,
    ranking_tokens_for_query,
    search_tokens_for_metadata,
)
from app.services.product_query_normalizer import normalize_product_query

load_dotenv()

OUT = Path(__file__).resolve().parent / "results" / "latency_explain.json"


def explain(db, sql: str, params: dict) -> str:
    rows = db.execute(text("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) " + sql), params)
    return "\n".join(row[0] for row in rows)


def main() -> None:
    db = SessionLocal()
    report: dict = {"product": {}, "orders": {}, "indexes": {}}
    try:
        query = "کرم ضد چروک پرایم"
        metadata = extract_product_metadata(query)
        tokens = search_tokens_for_metadata(query, metadata)
        sql, params = build_product_search_sql(
            query=normalize_product_query(query),
            tokens=tokens,
            filters=metadata,
            ranking_tokens=ranking_tokens_for_query(query),
            rank_query=product_name_phrase(query, metadata) or ranking_query_text(query),
            name_tokens=product_name_tokens(query, metadata),
            name_phrase=product_name_phrase(query, metadata),
            limit=10,
        )
        report["product"]["query"] = query
        report["product"]["has_sql"] = bool(sql)
        if sql:
            report["product"]["explain"] = explain(db, sql, params)

        report["orders"]["summary"] = explain(
            db,
            """
            SELECT COUNT(DISTINCT order_id) AS total_orders
            FROM public.orders
            WHERE customer_id = :customer_id
            """,
            {"customer_id": "9206288"},
        )
        report["indexes"]["products"] = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public' AND tablename = 'products'
                    ORDER BY indexname
                    """
                )
            ).mappings()
        ]
        report["indexes"]["orders"] = [
            dict(row)
            for row in db.execute(
                text(
                    """
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE schemaname = 'public' AND tablename = 'orders'
                    ORDER BY indexname
                    """
                )
            ).mappings()
        ]
        report["stats"] = dict(
            db.execute(
                text(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM public.products) AS products,
                        (SELECT COUNT(*) FROM public.orders) AS orders
                    """
                )
            ).mappings().one()
        )
    finally:
        db.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stats", "indexes")}, ensure_ascii=False, indent=2))
    print("\n--- PRODUCT EXPLAIN ---\n")
    print(report["product"].get("explain", "no sql"))
    print("\n--- ORDER EXPLAIN ---\n")
    print(report["orders"]["summary"])


if __name__ == "__main__":
    main()
