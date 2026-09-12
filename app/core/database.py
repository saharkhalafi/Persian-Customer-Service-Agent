import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.core.config import (
    DB_CONNECT_TIMEOUT_SECONDS,
    DB_MAX_OVERFLOW,
    DB_POOL_RECYCLE_SECONDS,
    DB_POOL_SIZE,
    DB_POOL_TIMEOUT_SECONDS,
    DB_READY_TIMEOUT_SECONDS,
    DB_STATEMENT_TIMEOUT_MS,
)


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set in environment variables."
    )


def _connect_args(url: str) -> dict:
    if not str(url).startswith("postgres"):
        return {}
    return {
        "connect_timeout": DB_CONNECT_TIMEOUT_SECONDS,
        "options": f"-c statement_timeout={DB_STATEMENT_TIMEOUT_MS}",
    }


ENGINE_CONNECT_ARGS = _connect_args(DATABASE_URL)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT_SECONDS,
    pool_recycle=DB_POOL_RECYCLE_SECONDS,
    pool_reset_on_return="rollback",
    connect_args=ENGINE_CONNECT_ARGS,
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def ensure_indexes() -> None:
    if not str(DATABASE_URL).startswith("postgres"):
        return
    statements = (
        "CREATE INDEX IF NOT EXISTS idx_products_brand_lower "
        "ON public.products (LOWER(TRIM(brand)))",
    )
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


try:
    ensure_indexes()
except Exception:
    pass


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def check_database(timeout_seconds: float | None = None) -> bool:
    limit = DB_READY_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds

    def _ping() -> None:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_ping)
        try:
            future.result(timeout=limit)
        except (FuturesTimeout, Exception):
            future.cancel()
            return False
    return True
