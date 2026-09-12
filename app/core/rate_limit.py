from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Depends

from app.core.config import CHAT_RATE_LIMIT_REQUESTS, CHAT_RATE_LIMIT_WINDOW_SECONDS
from app.core.auth import get_request_context
from app.core.context import RequestContext
from app.core.exceptions import RateLimitError


class SlidingWindowLimiter:
    """In-process limiter. Multi-instance production should use a shared backend."""

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        max_keys: int = 10_000,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            queue = self._events.setdefault(key, deque())
            while queue and queue[0] <= cutoff:
                queue.popleft()
            if len(queue) >= self.max_requests:
                return False
            queue.append(now)
            if len(self._events) > self.max_keys:
                self._evict_stale(cutoff)
            return True

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _evict_stale(self, cutoff: float) -> None:
        stale = [
            key
            for key, queue in self._events.items()
            if not queue or queue[-1] <= cutoff
        ]
        for key in stale:
            self._events.pop(key, None)
        while len(self._events) > self.max_keys:
            self._events.pop(next(iter(self._events)))


_chat_limiter = SlidingWindowLimiter(
    max_requests=CHAT_RATE_LIMIT_REQUESTS,
    window_seconds=CHAT_RATE_LIMIT_WINDOW_SECONDS,
)


def get_chat_limiter() -> SlidingWindowLimiter:
    return _chat_limiter


def reset_chat_limiter() -> None:
    _chat_limiter.reset()


def enforce_chat_rate_limit(
    context: RequestContext = Depends(get_request_context),
) -> RequestContext:
    if not get_chat_limiter().allow(f"customer:{context.customer_id}"):
        raise RateLimitError()
    return context
