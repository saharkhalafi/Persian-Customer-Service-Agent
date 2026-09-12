from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class TtlLruCache:
    def __init__(self, maxsize: int = 256, ttl_seconds: float = 3600.0):
        self.maxsize = max(1, maxsize)
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expires_at, value = item
            if expires_at < now:
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: Hashable, value: Any) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        with self._lock:
            self._data[key] = (expires_at, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0
