from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    try:
        parsed = int(raw) if raw is not None and str(raw).strip() else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        parsed = float(raw) if raw is not None and str(raw).strip() else default
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def env_csv(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


# Gemini / LLM
GEMINI_TIMEOUT_MS = env_int("GEMINI_TIMEOUT_MS", 45_000, 5_000, 120_000)
GEMINI_RETRY_ATTEMPTS = env_int("GEMINI_RETRY_ATTEMPTS", 2, 1, 3)

# PostgreSQL
DB_CONNECT_TIMEOUT_SECONDS = env_int("DB_CONNECT_TIMEOUT_SECONDS", 5, 1, 30)
DB_STATEMENT_TIMEOUT_MS = env_int("DB_STATEMENT_TIMEOUT_MS", 30_000, 1_000, 120_000)
DB_POOL_SIZE = env_int("DB_POOL_SIZE", 10, 1, 50)
DB_MAX_OVERFLOW = env_int("DB_MAX_OVERFLOW", 20, 0, 50)
DB_POOL_TIMEOUT_SECONDS = env_int("DB_POOL_TIMEOUT_SECONDS", 10, 1, 60)
DB_POOL_RECYCLE_SECONDS = env_int("DB_POOL_RECYCLE_SECONDS", 1_800, 60, 7_200)
DB_READY_TIMEOUT_SECONDS = env_float("DB_READY_TIMEOUT_SECONDS", 2.0, 0.2, 10.0)

# Chat API
CHAT_RATE_LIMIT_REQUESTS = env_int("CHAT_RATE_LIMIT_REQUESTS", 20, 1, 1_000)
CHAT_RATE_LIMIT_WINDOW_SECONDS = env_int("CHAT_RATE_LIMIT_WINDOW_SECONDS", 60, 1, 3_600)
CHAT_MAX_BODY_BYTES = env_int("CHAT_MAX_BODY_BYTES", 16_384, 1_024, 1_048_576)
CONVERSATION_HISTORY_LIMIT = env_int("CONVERSATION_HISTORY_LIMIT", 8, 0, 20)

# CORS: local Streamlit/Swagger by default. Production should set an explicit list.
CORS_ALLOWED_ORIGINS = env_csv(
    "CORS_ALLOWED_ORIGINS",
    "http://localhost:8000,http://localhost:8501,http://127.0.0.1:8000,http://127.0.0.1:8501",
)
