"""Small process-local caches and latency history for chat requests."""

from __future__ import annotations

import copy
import re
import threading
import time
from collections import OrderedDict, deque
from typing import Any


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.casefold()).strip()


class AnswerCache:
    def __init__(self, maxsize: int = 256, ttl_seconds: int = 24 * 60 * 60):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, query: str) -> dict[str, Any] | None:
        key = normalize_query(query)
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, query: str, value: dict[str, Any]) -> None:
        key = normalize_query(query)
        if not key:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + self.ttl_seconds, copy.deepcopy(value))
            self._items.move_to_end(key)
            while len(self._items) > self.maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class LatencyHistory:
    """Store only timing data; never retain prompts, answers, or source text."""

    def __init__(self, max_records: int = 50):
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def add(self, response_time_ms: float | None, timing_breakdown: list[dict[str, Any]]) -> None:
        with self._lock:
            self._records.append({
                "timestamp": time.time(),
                "response_time_ms": response_time_ms,
                "timing_breakdown": copy.deepcopy(timing_breakdown),
            })

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(list(self._records))


answer_cache = AnswerCache()
latency_history = LatencyHistory()

PREWARM_CACHE_QUERIES = {
    normalize_query("What should I do about high fever lasting 3 days?"),
    normalize_query("I've had a persistent cough for 3 weeks"),
    normalize_query("What are early warning signs and home care for Dengue fever?"),
    normalize_query("What antenatal checkups and nutrition are needed during pregnancy?"),
    normalize_query("Do antibiotics cure viral infections like cold or flu?"),
    normalize_query("Does turmeric cure diabetes?"),
}
