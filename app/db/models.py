"""SQLAlchemy metadata for Alembic.

Repositories keep using raw SQL. These models exist so schema changes are
migrated explicitly and are not created on application startup.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, MetaData, Text, text
from sqlalchemy.orm import DeclarativeBase


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("idx_users_customer_code", "customer_code"),
        {"schema": "public"},
    )

    customer_code = Column(Text, primary_key=True)
    customer_group = Column(Text)
    customer_type = Column(Text)
    city = Column(Text)
    personal_code = Column(Text)
    customer_name = Column(Text)
    gender = Column(Text)
    mobile = Column(Text)
    email = Column(Text)
    created_date_persian = Column(Text)
    first_purchase = Column(Text)
    last_purchase = Column(Text)
    success_ordered_cnt = Column(Text)
    success_order_price = Column(Text)
    item_cnt = Column(Text)


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        Index("idx_products_product_code", "product_code"),
        Index("idx_products_brand", "brand"),
        {"schema": "public"},
    )

    product_code = Column(Text, primary_key=True)
    product_name = Column(Text)
    brand = Column(Text)
    type = Column(Text)
    sku = Column(Text)
    category_level1 = Column(Text)
    category_level2 = Column(Text)
    last_status = Column(Text)
    available_qty = Column(Text)
    special_price = Column(Text)
    sum_of_price = Column(Text)
    color = Column(Text)
    size = Column(Text)
    gender = Column(Text)


class Order(Base):
    __tablename__ = "orders"
    __table_args__ = (
        Index("idx_orders_customer_id", "customer_id"),
        Index("idx_orders_order_id", "order_id"),
        {"schema": "public"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(Text, nullable=False)
    order_id = Column(Text)
    created_at = Column(Text)
    status = Column(Text)
    delivered_date = Column(Text)
    method = Column(Text)
    coupon_code = Column(Text)
    delivery_time = Column(Text)
    city = Column(Text)
    seller_id = Column(Text)
    seller_type = Column(Text)
    product_id = Column(Text)
    order_items_name = Column(Text)
    order_items_sku = Column(Text)
    brand_name = Column(Text)
    category_level1 = Column(Text)
    category_level2 = Column(Text)
    invoiced_qty = Column(Text)
    ordered_qty = Column(Text)
    refund_qty = Column(Text)
    shipped_qty = Column(Text)
    sale_price = Column(Text)
    final_price = Column(Text)
    color = Column(Text)
    material = Column(Text)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        Index(
            "idx_conversation_messages_customer_created",
            "customer_id",
            text("created_at DESC"),
        ),
        {"schema": "public"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(Text, nullable=False)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (
        Index(
            "idx_message_feedback_customer_created",
            "customer_id",
            text("created_at DESC"),
        ),
        {"schema": "public"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_id = Column(Text, nullable=False)
    conversation_id = Column(Text)
    message_id = Column(Text)
    rating = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    )
