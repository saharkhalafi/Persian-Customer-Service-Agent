"""Initial application schema.

Creates catalog tables (users, products, orders) when missing so a fresh
environment can start, and creates application-owned tables
conversation_messages and message_feedback. Existing production tables
are left unchanged.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-12
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect, text

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "public"


def _inspector():
    return inspect(op.get_bind())


def _table_names() -> set[str]:
    return set(_inspector().get_table_names(schema=SCHEMA))


def _index_names(table: str) -> set[str]:
    if table not in _table_names():
        return set()
    return {item["name"] for item in _inspector().get_indexes(table, schema=SCHEMA)}


def _create_index_if_missing(name: str, table: str, columns: list[str]) -> None:
    if name in _index_names(table):
        return
    op.create_index(name, table, columns, schema=SCHEMA)


def upgrade() -> None:
    tables = _table_names()

    if "users" not in tables:
        op.create_table(
            "users",
            sa.Column("customer_code", sa.Text(), nullable=False),
            sa.Column("customer_group", sa.Text(), nullable=True),
            sa.Column("customer_type", sa.Text(), nullable=True),
            sa.Column("city", sa.Text(), nullable=True),
            sa.Column("personal_code", sa.Text(), nullable=True),
            sa.Column("customer_name", sa.Text(), nullable=True),
            sa.Column("gender", sa.Text(), nullable=True),
            sa.Column("mobile", sa.Text(), nullable=True),
            sa.Column("email", sa.Text(), nullable=True),
            sa.Column("created_date_persian", sa.Text(), nullable=True),
            sa.Column("first_purchase", sa.Text(), nullable=True),
            sa.Column("last_purchase", sa.Text(), nullable=True),
            sa.Column("success_ordered_cnt", sa.Text(), nullable=True),
            sa.Column("success_order_price", sa.Text(), nullable=True),
            sa.Column("item_cnt", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("customer_code", name="pk_users"),
            schema=SCHEMA,
        )
    _create_index_if_missing("idx_users_customer_code", "users", ["customer_code"])

    if "products" not in _table_names():
        op.create_table(
            "products",
            sa.Column("product_code", sa.Text(), nullable=False),
            sa.Column("product_name", sa.Text(), nullable=True),
            sa.Column("brand", sa.Text(), nullable=True),
            sa.Column("type", sa.Text(), nullable=True),
            sa.Column("sku", sa.Text(), nullable=True),
            sa.Column("category_level1", sa.Text(), nullable=True),
            sa.Column("category_level2", sa.Text(), nullable=True),
            sa.Column("last_status", sa.Text(), nullable=True),
            sa.Column("available_qty", sa.Text(), nullable=True),
            sa.Column("special_price", sa.Text(), nullable=True),
            sa.Column("sum_of_price", sa.Text(), nullable=True),
            sa.Column("color", sa.Text(), nullable=True),
            sa.Column("size", sa.Text(), nullable=True),
            sa.Column("gender", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("product_code", name="pk_products"),
            schema=SCHEMA,
        )
    _create_index_if_missing("idx_products_product_code", "products", ["product_code"])
    _create_index_if_missing("idx_products_brand", "products", ["brand"])
    if "idx_products_brand_lower" not in _index_names("products"):
        op.execute(
            text(
                "CREATE INDEX idx_products_brand_lower "
                "ON public.products (LOWER(TRIM(brand)))"
            )
        )

    if "orders" not in _table_names():
        op.create_table(
            "orders",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("customer_id", sa.Text(), nullable=False),
            sa.Column("order_id", sa.Text(), nullable=True),
            sa.Column("created_at", sa.Text(), nullable=True),
            sa.Column("status", sa.Text(), nullable=True),
            sa.Column("delivered_date", sa.Text(), nullable=True),
            sa.Column("method", sa.Text(), nullable=True),
            sa.Column("coupon_code", sa.Text(), nullable=True),
            sa.Column("delivery_time", sa.Text(), nullable=True),
            sa.Column("city", sa.Text(), nullable=True),
            sa.Column("seller_id", sa.Text(), nullable=True),
            sa.Column("seller_type", sa.Text(), nullable=True),
            sa.Column("product_id", sa.Text(), nullable=True),
            sa.Column("order_items_name", sa.Text(), nullable=True),
            sa.Column("order_items_sku", sa.Text(), nullable=True),
            sa.Column("brand_name", sa.Text(), nullable=True),
            sa.Column("category_level1", sa.Text(), nullable=True),
            sa.Column("category_level2", sa.Text(), nullable=True),
            sa.Column("invoiced_qty", sa.Text(), nullable=True),
            sa.Column("ordered_qty", sa.Text(), nullable=True),
            sa.Column("refund_qty", sa.Text(), nullable=True),
            sa.Column("shipped_qty", sa.Text(), nullable=True),
            sa.Column("sale_price", sa.Text(), nullable=True),
            sa.Column("final_price", sa.Text(), nullable=True),
            sa.Column("color", sa.Text(), nullable=True),
            sa.Column("material", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id", name="pk_orders"),
            schema=SCHEMA,
        )
    _create_index_if_missing("idx_orders_customer_id", "orders", ["customer_id"])
    _create_index_if_missing("idx_orders_order_id", "orders", ["order_id"])

    if "conversation_messages" not in _table_names():
        op.create_table(
            "conversation_messages",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("customer_id", sa.Text(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("NOW()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id", name="pk_conversation_messages"),
            schema=SCHEMA,
        )
    if "idx_conversation_messages_customer_created" not in _index_names(
        "conversation_messages"
    ):
        op.execute(
            text(
                "CREATE INDEX idx_conversation_messages_customer_created "
                "ON public.conversation_messages (customer_id, created_at DESC)"
            )
        )

    if "message_feedback" not in _table_names():
        op.create_table(
            "message_feedback",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("customer_id", sa.Text(), nullable=False),
            sa.Column("conversation_id", sa.Text(), nullable=True),
            sa.Column("message_id", sa.Text(), nullable=True),
            sa.Column("rating", sa.Text(), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("NOW()"),
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id", name="pk_message_feedback"),
            schema=SCHEMA,
        )
    if "idx_message_feedback_customer_created" not in _index_names("message_feedback"):
        op.execute(
            text(
                "CREATE INDEX idx_message_feedback_customer_created "
                "ON public.message_feedback (customer_id, created_at DESC)"
            )
        )


def downgrade() -> None:
    """Drop only application-owned objects. Catalog tables stay in place."""
    op.execute(text("DROP INDEX IF EXISTS public.idx_message_feedback_customer_created"))
    op.execute(
        text("DROP INDEX IF EXISTS public.idx_conversation_messages_customer_created")
    )
    op.execute(text("DROP INDEX IF EXISTS public.idx_products_brand_lower"))
    if "message_feedback" in _table_names():
        op.drop_table("message_feedback", schema=SCHEMA)
    if "conversation_messages" in _table_names():
        op.drop_table("conversation_messages", schema=SCHEMA)
