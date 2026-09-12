"""Standalone tool-calling evaluation.

Runs every question in tool_eval_data.json through the real Agent.
Does not score answer quality — only which tools were called.

Usage (from repo root):
    python -m evaluation.tool_eval
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.api.dependencies import get_agent, get_gemini_client
from app.core.context import RequestContext
from app.core.database import SessionLocal


DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "tool_eval_data.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "tool_eval_results.json"

CUSTOMER_ID = "9206288"
MAX_ATTEMPTS = 3
RETRY_SLEEP_SEC = 2.0


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")

    return data


def is_connection_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    text = str(exc)
    markers = (
        "10061",
        "10053",
        "10054",
        "actively refused",
        "Connection refused",
        "ConnectError",
        "ReadError",
        "ConnectTimeout",
        "TimeoutException",
    )
    if any(marker in text for marker in markers):
        return True
    return name in {
        "ConnectError",
        "ReadError",
        "ConnectTimeout",
        "TimeoutException",
        "RemoteProtocolError",
    }


def run_agent(agent, context: RequestContext, question: str):
    last_error: BaseException | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return agent.run(
                user_message=question,
                context=context,
            )
        except Exception as exc:
            last_error = exc
            if attempt < MAX_ATTEMPTS and is_connection_error(exc):
                print(
                    f"  connection error (attempt {attempt}/{MAX_ATTEMPTS}): {exc}"
                )
                time.sleep(RETRY_SLEEP_SEC * attempt)
                continue
            raise

    raise last_error or RuntimeError("agent.run failed")


def extract_tool_names(result) -> list[str]:
    names: list[str] = []
    for call in getattr(result, "tool_calls", None) or []:
        names.append(getattr(call, "name", str(call)))
    return names


def is_refusal_case(item: dict) -> bool:
    behavior = str(item.get("expected_behavior") or "").strip().lower()
    expected_tools = list(item.get("expected_tools") or [])
    return behavior == "refuse" or len(expected_tools) == 0


def evaluate_selection(expected_tools: list[str], actual_tools: list[str]) -> bool:
    return actual_tools == expected_tools


def pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def main() -> None:
    dataset = load_dataset()

    print("=" * 70)
    print("TOOL-CALLING EVALUATION")
    print("=" * 70)
    print(f"Dataset : {DATASET_PATH}")
    print(f"Samples : {len(dataset)}")
    print()

    db = SessionLocal()
    cases: list[dict] = []

    try:
        gemini = get_gemini_client()
        agent = get_agent(db=db, gemini=gemini)
        context = RequestContext(customer_id=CUSTOMER_ID)

        for index, item in enumerate(dataset, start=1):
            case_id = item.get("id")
            category = item.get("category")
            difficulty = item.get("difficulty")
            question = item["question"]
            expected_behavior = item.get("expected_behavior")
            expected_tools = list(item.get("expected_tools") or [])
            expected_tool_count = int(item.get("expected_tool_count") or 0)

            print(f"[{index}/{len(dataset)}] {case_id} {question}")

            error = None
            actual_tools: list[str] = []

            try:
                result = run_agent(agent, context, question)
                actual_tools = extract_tool_names(result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR: {error}")

            if is_refusal_case(item):
                expected_tools = []
                expected_tool_count = 0

            actual_tool_count = len(actual_tools)
            selection_pass = evaluate_selection(expected_tools, actual_tools)
            count_pass = actual_tool_count == expected_tool_count
            passed = selection_pass and count_pass and error is None
            verdict = "PASS" if passed else "FAIL"

            print(f"  expected: {expected_tools}")
            print(f"  actual  : {actual_tools}")
            print(f"  {verdict}")
            print()

            cases.append(
                {
                    "id": case_id,
                    "category": category,
                    "difficulty": difficulty,
                    "question": question,
                    "expected_tools": expected_tools,
                    "actual_tools": actual_tools,
                    "expected_tool_count": expected_tool_count,
                    "actual_tool_count": actual_tool_count,
                    "expected_behavior": expected_behavior,
                    "pass": passed,
                    "verdict": verdict,
                    "selection_pass": selection_pass,
                    "count_pass": count_pass,
                    "error": error,
                }
            )

    finally:
        db.close()

    total = len(cases)
    selection_ok = sum(1 for case in cases if case["selection_pass"])
    count_ok = sum(1 for case in cases if case["count_pass"])
    passed_ok = sum(1 for case in cases if case["pass"])

    by_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "total": 0}
    )
    by_tool: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "total": 0}
    )

    for case in cases:
        category = case["category"] or "unknown"
        by_category[category]["total"] += 1
        if case["pass"]:
            by_category[category]["passed"] += 1

        seen_expected = set()
        for tool_name in case["expected_tools"]:
            if tool_name in seen_expected:
                continue
            seen_expected.add(tool_name)
            by_tool[tool_name]["total"] += 1
            if tool_name in case["actual_tools"]:
                by_tool[tool_name]["passed"] += 1

    selection_accuracy = pct(selection_ok, total)
    count_accuracy = pct(count_ok, total)

    category_summary = {
        name: {
            "passed": stats["passed"],
            "total": stats["total"],
            "accuracy": round(pct(stats["passed"], stats["total"]), 2),
        }
        for name, stats in sorted(by_category.items())
    }
    tool_summary = {
        name: {
            "passed": stats["passed"],
            "total": stats["total"],
            "accuracy": round(pct(stats["passed"], stats["total"]), 2),
        }
        for name, stats in sorted(by_tool.items())
    }

    failed_cases = [
        {
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "expected_tools": case["expected_tools"],
            "actual_tools": case["actual_tools"],
            "error": case["error"],
        }
        for case in cases
        if not case["pass"]
    ]

    output = {
        "dataset": str(DATASET_PATH),
        "num_samples": total,
        "tool_selection_accuracy": round(selection_accuracy, 2),
        "tool_count_accuracy": round(count_accuracy, 2),
        "overall_passed": passed_ok,
        "overall_total": total,
        "by_category": category_summary,
        "by_tool": tool_summary,
        "failed_cases": failed_cases,
        "results": cases,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Tool Selection Accuracy : {selection_accuracy:.2f}%")
    print(f"Tool Count Accuracy     : {count_accuracy:.2f}%")
    print(f"Overall Passed          : {passed_ok} / {total}")
    print()
    print("By category:")
    for name, stats in category_summary.items():
        print(
            f"  {name}: {stats['accuracy']:.2f}% "
            f"({stats['passed']}/{stats['total']})"
        )
    print()
    print("By tool:")
    for name, stats in tool_summary.items():
        print(
            f"  {name}: {stats['accuracy']:.2f}% "
            f"({stats['passed']}/{stats['total']})"
        )

    if failed_cases:
        print()
        print("Failed cases:")
        for case in failed_cases:
            print(f"  {case['id']} [{case['category']}]")
            print(f"    expected: {case['expected_tools']}")
            print(f"    actual  : {case['actual_tools']}")
            if case["error"]:
                print(f"    error   : {case['error']}")

    print()
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
