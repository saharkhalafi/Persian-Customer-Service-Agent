from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


_LATENCY_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)


def _label_key(labels: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not labels:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in labels.items()))


class _Counter:
    def __init__(self) -> None:
        self._values: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        with self._lock:
            self._values[key] += amount

    def snapshot(self) -> dict[tuple[tuple[str, str], ...], float]:
        with self._lock:
            return dict(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


class _Histogram:
    def __init__(self, buckets: tuple[float, ...] = _LATENCY_BUCKETS_MS) -> None:
        self.buckets = buckets
        self._counts: dict[tuple[tuple[str, str], ...], list[int]] = {}
        self._sums: dict[tuple[tuple[str, str], ...], float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        key = _label_key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            placed = False
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    counts[index] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1
            self._sums[key] += value

    def snapshot(self) -> dict[tuple[tuple[str, str], ...], tuple[list[int], float]]:
        with self._lock:
            return {
                key: (list(counts), self._sums[key])
                for key, counts in self._counts.items()
            }

    def clear(self) -> None:
        with self._lock:
            self._counts.clear()
            self._sums.clear()


class MetricsRegistry:
    """In-process counters/histograms with Prometheus text export.

    No scrape stack is required. OpenTelemetry/Prometheus can later replace
    this registry without changing call sites.
    """

    def __init__(self) -> None:
        self.api_requests = _Counter()
        self.api_errors = _Counter()
        self.api_latency_ms = _Histogram()
        self.agent_requests = _Counter()
        self.agent_tool_calls = _Counter()
        self.agent_tool_selection_errors = _Counter()
        self.agent_latency_ms = _Histogram()
        self.product_search = _Counter()
        self.product_search_empty = _Counter()
        self.product_search_latency_ms = _Histogram()
        self.product_search_llm_fallback = _Counter()
        self.product_search_llm_fallback_failures = _Counter()
        self.llm_calls = _Counter()
        self.llm_failures = _Counter()
        self.llm_timeouts = _Counter()
        self.llm_latency_ms = _Histogram()
        self.llm_tokens = _Counter()
        self.llm_estimated_cost_usd = _Counter()
        self.feedback = _Counter()

    def reset(self) -> None:
        for value in self.__dict__.values():
            if hasattr(value, "clear"):
                value.clear()

    def render_prometheus(self) -> str:
        lines: list[str] = [
            "# HELP app_api_requests_total HTTP requests handled by the API",
            "# TYPE app_api_requests_total counter",
        ]
        lines.extend(_render_counter("app_api_requests_total", self.api_requests))
        lines.append("# TYPE app_api_errors_total counter")
        lines.extend(_render_counter("app_api_errors_total", self.api_errors))
        lines.extend(_render_histogram("app_api_request_latency_milliseconds", self.api_latency_ms))
        lines.append("# TYPE app_agent_requests_total counter")
        lines.extend(_render_counter("app_agent_requests_total", self.agent_requests))
        lines.append("# TYPE app_agent_tool_calls_total counter")
        lines.extend(_render_counter("app_agent_tool_calls_total", self.agent_tool_calls))
        lines.append("# TYPE app_agent_tool_selection_errors_total counter")
        lines.extend(
            _render_counter(
                "app_agent_tool_selection_errors_total",
                self.agent_tool_selection_errors,
            )
        )
        lines.extend(_render_histogram("app_agent_latency_milliseconds", self.agent_latency_ms))
        lines.append("# TYPE app_product_search_total counter")
        lines.extend(_render_counter("app_product_search_total", self.product_search))
        lines.append("# TYPE app_product_search_empty_total counter")
        lines.extend(_render_counter("app_product_search_empty_total", self.product_search_empty))
        lines.extend(
            _render_histogram(
                "app_product_search_latency_milliseconds",
                self.product_search_latency_ms,
            )
        )
        lines.append("# TYPE app_product_search_llm_fallback_total counter")
        lines.extend(
            _render_counter(
                "app_product_search_llm_fallback_total",
                self.product_search_llm_fallback,
            )
        )
        lines.append("# TYPE app_product_search_llm_fallback_failures_total counter")
        lines.extend(
            _render_counter(
                "app_product_search_llm_fallback_failures_total",
                self.product_search_llm_fallback_failures,
            )
        )
        lines.append("# TYPE app_llm_calls_total counter")
        lines.extend(_render_counter("app_llm_calls_total", self.llm_calls))
        lines.append("# TYPE app_llm_failures_total counter")
        lines.extend(_render_counter("app_llm_failures_total", self.llm_failures))
        lines.append("# TYPE app_llm_timeouts_total counter")
        lines.extend(_render_counter("app_llm_timeouts_total", self.llm_timeouts))
        lines.extend(_render_histogram("app_llm_latency_milliseconds", self.llm_latency_ms))
        lines.append("# TYPE app_llm_tokens_total counter")
        lines.extend(_render_counter("app_llm_tokens_total", self.llm_tokens))
        lines.append("# TYPE app_llm_estimated_cost_usd_total counter")
        lines.extend(
            _render_counter(
                "app_llm_estimated_cost_usd_total",
                self.llm_estimated_cost_usd,
            )
        )
        lines.append("# TYPE app_feedback_total counter")
        lines.extend(_render_counter("app_feedback_total", self.feedback))
        return "\n".join(lines) + "\n"


_SKIP_API_PATHS = {
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}

registry = MetricsRegistry()


def reset_metrics() -> None:
    registry.reset()


def record_from_event(event: str, fields: dict[str, Any]) -> None:
    if event == "request_completed":
        endpoint = str(fields.get("endpoint") or "")
        if endpoint in _SKIP_API_PATHS:
            return
        status = int(fields.get("status") or 0)
        labels = {
            "endpoint": endpoint,
            "method": str(fields.get("method") or ""),
            "status": str(status),
        }
        registry.api_requests.inc(labels=labels)
        latency = _as_float(fields.get("latency_ms"))
        if latency is not None:
            registry.api_latency_ms.observe(latency, labels={"endpoint": endpoint})
        if status >= 400:
            registry.api_errors.inc(
                labels={"endpoint": endpoint, "status": str(status)}
            )
        return

    if event == "agent_completed":
        registry.agent_requests.inc()
        latency = _as_float(fields.get("latency_ms"))
        if latency is not None:
            registry.agent_latency_ms.observe(latency)
        return

    if event == "tool_execution":
        registry.agent_tool_calls.inc(
            labels={
                "tool": str(fields.get("tool") or "unknown"),
                "status": str(fields.get("status") or "unknown"),
            }
        )
        return

    if event == "agent_tool_selection_error":
        registry.agent_tool_selection_errors.inc(
            labels={"tool": str(fields.get("tool") or "unknown")}
        )
        return

    if event == "llm_call":
        purpose = str(fields.get("purpose") or "unknown")
        status = str(fields.get("status") or "success")
        registry.llm_calls.inc(labels={"purpose": purpose, "status": status})
        latency = _as_float(fields.get("latency_ms"))
        if latency is not None:
            registry.llm_latency_ms.observe(latency, labels={"purpose": purpose})
        if status != "success":
            registry.llm_failures.inc(labels={"purpose": purpose})
        if fields.get("timeout"):
            registry.llm_timeouts.inc(labels={"purpose": purpose})
        input_tokens = _as_float(fields.get("input_tokens")) or 0.0
        output_tokens = _as_float(fields.get("output_tokens")) or 0.0
        if input_tokens:
            registry.llm_tokens.inc(input_tokens, labels={"direction": "input"})
        if output_tokens:
            registry.llm_tokens.inc(output_tokens, labels={"direction": "output"})
        cost = _as_float(fields.get("estimated_cost_usd"))
        if cost:
            registry.llm_estimated_cost_usd.inc(cost)
        return

    if event == "product_search_completed":
        registry.product_search.inc()
        latency = _as_float(fields.get("latency_ms"))
        if latency is not None:
            registry.product_search_latency_ms.observe(latency)
        if fields.get("empty"):
            registry.product_search_empty.inc()
        if fields.get("llm_fallback_triggered"):
            registry.product_search_llm_fallback.inc()
        if fields.get("llm_failure"):
            registry.product_search_llm_fallback_failures.inc()
        return

    if event == "feedback_recorded":
        registry.feedback.inc(labels={"rating": str(fields.get("rating") or "")})


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{key}="{value}"' for key, value in labels)
    return "{" + inner + "}"


def _render_counter(name: str, counter: _Counter) -> list[str]:
    lines = []
    snapshot = counter.snapshot()
    if not snapshot:
        lines.append(f"{name} 0")
        return lines
    for labels, value in snapshot.items():
        lines.append(f"{name}{_render_labels(labels)} {value}")
    return lines


def _render_histogram(name: str, histogram: _Histogram) -> list[str]:
    lines = [
        f"# TYPE {name} histogram",
    ]
    snapshot = histogram.snapshot()
    if not snapshot:
        return lines
    for labels, (counts, total) in snapshot.items():
        cumulative = 0
        for index, bound in enumerate(histogram.buckets):
            cumulative += counts[index]
            bucket_labels = labels + (("le", _format_le(bound)),)
            lines.append(f"{name}_bucket{_render_labels(bucket_labels)} {cumulative}")
        cumulative += counts[-1]
        inf_labels = labels + (("le", "+Inf"),)
        lines.append(f"{name}_bucket{_render_labels(inf_labels)} {cumulative}")
        lines.append(f"{name}_sum{_render_labels(labels)} {total}")
        lines.append(f"{name}_count{_render_labels(labels)} {cumulative}")
    return lines


def _format_le(bound: float) -> str:
    if bound == int(bound):
        return str(int(bound))
    return str(bound)
