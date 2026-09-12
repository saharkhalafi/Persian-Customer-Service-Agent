from __future__ import annotations

import re

from app.core.exceptions import AuthenticationError, ValidationFailedError

CUSTOMER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_UNSAFE_TOOL_KEYS = {"customer_id", "sql", "filters"}
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|password|secret|token|database_url|gemini_api_key)"
    r"(\s*[=:]\s*)(\S+)"
)


def validate_customer_id(value: str | None) -> str:
    customer_id = str(value or "").strip()
    if not customer_id or not CUSTOMER_ID_PATTERN.fullmatch(customer_id):
        raise AuthenticationError("Customer authentication is required")
    return customer_id


def sanitize_tool_arguments(args: dict | None) -> dict:
    cleaned: dict = {}
    for key, value in dict(args or {}).items():
        if str(key).strip().lower() in _UNSAFE_TOOL_KEYS:
            continue
        cleaned[key] = value
    return cleaned


def sanitize_short_id(value: object, maximum: int = 64) -> str:
    text = str(value or "").strip()[:maximum]
    if not text or any(ch in text for ch in ";'\"\\\n\r\x00"):
        return ""
    return text


def enforce_content_length(content_length: str | None, maximum: int) -> None:
    if not content_length:
        return
    try:
        size = int(content_length)
    except (TypeError, ValueError) as exc:
        raise ValidationFailedError("Invalid Content-Length") from exc
    if size > maximum:
        raise ValidationFailedError("Request body is too large")


def redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub(r"\1\2[REDACTED]", text)
