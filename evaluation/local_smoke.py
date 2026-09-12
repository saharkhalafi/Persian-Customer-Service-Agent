"""Optional live smoke against a running local API.

Usage (after uvicorn is up):

    set RUN_LIVE_SMOKE=1
    python -m evaluation.local_smoke

Does not change Product Search or Agent behavior. Failures are reported;
this script never rewrites benchmarks.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from evaluation.api_smoke_cases import SECURITY_SMOKE_CASES, SMOKE_CASES

BASE_URL = os.getenv("SMOKE_BASE_URL", "http://127.0.0.1:8000")
CUSTOMER_ID = os.getenv("SMOKE_CUSTOMER_ID", "9206288")


def _request(method: str, path: str, payload: dict | None = None, headers: dict | None = None):
    body = None
    request_headers = {"X-Customer-ID": CUSTOMER_ID}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw}
        return exc.code, parsed


def main() -> int:
    failures = []
    for path in ("/health", "/ready", "/metrics"):
        status, _payload = _request("GET", path)
        if path == "/metrics":
            ok = status == 200
        elif path == "/health":
            ok = status == 200
        else:
            ok = status in {200, 503}
        print(f"{path}: {status}")
        if not ok:
            failures.append(path)

    for case in (*SMOKE_CASES, *SECURITY_SMOKE_CASES):
        status, body = _request(
            "POST",
            "/api/v1/chat",
            {"message": case.message, "customer_id": "attacker-9999"},
            {"X-Trace-ID": f"live-{case.name}"},
        )
        tools = [item.get("name") for item in body.get("tool_calls", [])]
        print(f"{case.name}: {status} tools={tools} trace={body.get('trace_id')}")
        if status != 200:
            failures.append(case.name)
            continue
        if "error" in body:
            failures.append(case.name)
        if case.category == "security":
            dumped = json.dumps(body, ensure_ascii=False)
            if "DROP TABLE" in dumped or "attacker-9999" in dumped:
                failures.append(f"{case.name}: leaked unsafe fields")

    status, body = _request(
        "POST",
        "/api/v1/feedback",
        {"rating": "positive", "conversation_id": "live-smoke"},
    )
    print(f"feedback: {status} {body}")
    if status != 200:
        failures.append("feedback")

    if failures:
        print("FAILED:", failures)
        return 1
    print("LIVE SMOKE PASSED")
    return 0


if __name__ == "__main__":
    if os.getenv("RUN_LIVE_SMOKE", "").strip() not in {"1", "true", "yes", "on"}:
        print("Set RUN_LIVE_SMOKE=1 to hit a running API.")
        sys.exit(0)
    raise SystemExit(main())
