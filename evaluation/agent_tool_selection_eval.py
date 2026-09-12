"""Isolated Agent tool-selection and multi-tool evaluation.

Extends the existing tool_eval runner. Does not replace it.
Does not change Product Search or production Agent behavior.

Usage (from repo root):
    python -m evaluation.agent_tool_selection_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.api.dependencies import get_agent, get_gemini_client
from app.core.context import RequestContext
from app.core.database import SessionLocal
from evaluation.agent_tool_selection_metrics import (
    aggregate_results,
    compare_with_previous,
    render_report,
    score_case,
    unique_names,
    verdict_from_metrics,
)
from evaluation.tool_eval import CUSTOMER_ID, extract_tool_names, run_agent


DATASET_PATH = (
    PROJECT_ROOT / "evaluation" / "datasets" / "agent_tool_selection_eval.json"
)
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_PATH = RESULTS_DIR / "agent_tool_selection_eval.json"
REPORT_PATH = RESULTS_DIR / "agent_tool_selection_report.md"
PREVIOUS_RESULTS_PATH = RESULTS_DIR / "tool_eval_results.json"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def instrument_request_context(agent, recorder: list[dict]) -> None:
    """Record customer_id actually received by existing tools. Eval-only."""

    def wrap_method(owner, method_name: str) -> None:
        original = getattr(owner, method_name, None)
        if original is None or not callable(original):
            return

        @wraps(original)
        def wrapped(*args, **kwargs):
            context = kwargs.get("context")
            if context is None:
                for value in args:
                    if hasattr(value, "customer_id"):
                        context = value
                        break
            if context is not None:
                recorder.append(
                    {
                        "tool": method_name,
                        "customer_id": getattr(context, "customer_id", None),
                    }
                )
            return original(*args, **kwargs)

        setattr(owner, method_name, wrapped)

    for tools in (
        agent.order_tools,
        agent.customer_tools,
        agent.knowledge_tools,
        agent.conversation_tools,
    ):
        for name in dir(tools):
            if name.startswith("_"):
                continue
            attr = getattr(tools, name)
            if callable(attr):
                wrap_method(tools, name)


def extract_tool_calls(result) -> list[dict]:
    calls = []
    for call in getattr(result, "tool_calls", None) or []:
        calls.append(
            {
                "name": getattr(call, "name", ""),
                "arguments": dict(getattr(call, "arguments", None) or {}),
            }
        )
    return calls


def evaluate_offline(dataset: list[dict], predictions: list[dict]) -> dict:
    by_id = {item["id"]: item for item in dataset}
    cases = []
    for prediction in predictions:
        item = by_id[prediction["id"]]
        scored = score_case(
            item=item,
            actual_tools=list(prediction.get("actual_tools") or []),
            actual_calls=list(prediction.get("actual_calls") or []),
            answer=str(prediction.get("answer") or ""),
            scoped_customer_ids=list(
                prediction.get("scoped_customer_ids") or []
            ),
            expected_customer_id=prediction.get("expected_customer_id"),
        )
        cases.append(
            {
                **item,
                **scored,
                "actual_tools": list(prediction.get("actual_tools") or []),
                "actual_calls": list(prediction.get("actual_calls") or []),
                "answer": prediction.get("answer"),
            }
        )
    return build_payload(cases)


def build_payload(cases: list[dict]) -> dict:
    metrics = aggregate_results(cases)
    failed_cases = [
        {
            "id": case.get("id"),
            "category": case.get("category"),
            "question": case.get("question"),
            "expected_tools": case.get("expected_tools") or [],
            "actual_tools": case.get("actual_tools") or [],
            "failure_reason": case.get("failure_reason"),
        }
        for case in cases
        if not case.get("pass")
    ]
    return {
        "dataset": str(DATASET_PATH),
        "previous_eval": str(PREVIOUS_RESULTS_PATH),
        "metrics": metrics,
        "comparison": compare_with_previous(metrics),
        "verdict": verdict_from_metrics(metrics),
        "failed_cases": failed_cases,
        "results": cases,
    }


def write_outputs(payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    REPORT_PATH.write_text(render_report(payload), encoding="utf-8")


def _print(message: str) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write(
            (message + "\n").encode(encoding, errors="replace")
        )
        sys.stdout.buffer.flush()


def print_summary(payload: dict) -> None:
    _print(render_report(payload))
    _print(f"JSON saved to: {RESULTS_PATH}")
    _print(f"Report saved to: {REPORT_PATH}")


def run_live(dataset: list[dict]) -> dict:
    db = SessionLocal()
    cases: list[dict] = []
    try:
        gemini = get_gemini_client()
        agent = get_agent(db=db, gemini=gemini)
        context = RequestContext(customer_id=CUSTOMER_ID)
        scoped: list[dict] = []
        instrument_request_context(agent, scoped)

        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            _print(f"[{index}/{len(dataset)}] {item['id']} {question}")
            scoped.clear()
            error = None
            actual_tools: list[str] = []
            actual_calls: list[dict] = []
            answer = ""
            try:
                result = run_agent(agent, context, question)
                actual_calls = extract_tool_calls(result)
                actual_tools = unique_names(extract_tool_names(result))
                answer = getattr(result, "answer", "") or ""
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _print(f"  ERROR: {error}")

            scoped_ids = [
                row["customer_id"]
                for row in scoped
                if row.get("customer_id") is not None
            ]
            scored = score_case(
                item=item,
                actual_tools=actual_tools,
                actual_calls=actual_calls,
                answer=answer,
                scoped_customer_ids=scoped_ids,
                expected_customer_id=CUSTOMER_ID,
            )
            if error:
                scored["pass"] = False
                scored["failure_reason"] = error

            _print(f"  expected: {item.get('expected_tools') or []}")
            _print(f"  actual  : {actual_tools}")
            _print(f"  {'PASS' if scored['pass'] else 'FAIL'}")
            if scored.get("failure_reason"):
                _print(f"  reason  : {scored['failure_reason']}")
            _print("")

            cases.append(
                {
                    **item,
                    **scored,
                    "actual_tools": actual_tools,
                    "actual_calls": actual_calls,
                    "answer": answer,
                    "error": error,
                    "scoped_customer_ids": scoped_ids,
                }
            )
    finally:
        db.close()
    return build_payload(cases)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated agent tool-selection evaluation"
    )
    parser.add_argument(
        "--offline-predictions",
        help="Score a JSON list of predictions without calling Gemini",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Re-score the latest results JSON without calling Gemini",
    )
    args = parser.parse_args()
    dataset = load_dataset()

    if args.rescore:
        previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        predictions = [
            {
                "id": case["id"],
                "actual_tools": case.get("actual_tools") or [],
                "actual_calls": case.get("actual_calls") or [],
                "answer": case.get("answer") or "",
                "scoped_customer_ids": case.get("scoped_customer_ids") or [],
                "expected_customer_id": CUSTOMER_ID,
            }
            for case in previous.get("results") or []
        ]
        payload = evaluate_offline(dataset, predictions)
    elif args.offline_predictions:
        predictions = json.loads(
            Path(args.offline_predictions).read_text(encoding="utf-8")
        )
        payload = evaluate_offline(dataset, predictions)
    else:
        payload = run_live(dataset)

    write_outputs(payload)
    print_summary(payload)


if __name__ == "__main__":
    main()
