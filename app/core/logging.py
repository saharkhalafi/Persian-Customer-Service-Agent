from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

from app.core.observability import get_request_id, get_trace_id
from app.core.security import redact_secrets


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }
        trace_id = get_trace_id()
        request_id = get_request_id()
        if trace_id:
            payload["trace_id"] = trace_id
        if request_id:
            payload["request_id"] = request_id

        message = record.getMessage()
        if isinstance(record.msg, dict):
            payload.update(record.msg)
        else:
            try:
                parsed = json.loads(message)
            except (TypeError, ValueError):
                payload["message"] = message
            else:
                if isinstance(parsed, dict):
                    payload.update(parsed)
                else:
                    payload["message"] = message

        if record.exc_info:
            payload["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
        return redact_secrets(json.dumps(payload, ensure_ascii=False, default=str))


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root = logging.getLogger()
    if getattr(root, "_modiseh_json_logging", False):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    setattr(root, "_modiseh_json_logging", True)
