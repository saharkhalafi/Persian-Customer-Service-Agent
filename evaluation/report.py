import json
from pathlib import Path

from evaluation.metrics import (
    tool_selection_metrics,
    argument_exact_match,
    confusion_matrix,
    guardrail_metrics,
    latency_metrics,
)


def build_report(results: list[dict]) -> dict:

    intent_results = [
        r for r in results
        if r["suite"] == "intent_router"
    ]

    tool_results = [
        r for r in results
        if r["suite"] == "tool_calling"
    ]

    guardrail_results = [
        r for r in results
        if r["suite"] == "guardrails"
    ]

    return {
        "summary": {
            "total_cases": len(results),
            "failed_cases": sum(
                1 for r in results if r.get("error")
            ),
        },

        "intent_router": {
            "tool_metrics": tool_selection_metrics(
                intent_results
            ),
            "confusion_matrix": confusion_matrix(
                intent_results
            ),
        },

        "tool_calling": {
            "tool_metrics": tool_selection_metrics(
                tool_results
            ),
            "argument_metrics": argument_exact_match(
                tool_results
            ),
            "confusion_matrix": confusion_matrix(
                tool_results
            ),
        },

        "guardrails": guardrail_metrics(
            guardrail_results
        ),

        "latency": latency_metrics(results),
    }


def save_report(
    results: list[dict],
    output_dir: str = "evaluation/results",
):

    output = Path(output_dir)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_report(results)

    with open(
        output / "results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        output / "report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    failed = [
        r for r in results
        if r.get("error")
    ]

    with open(
        output / "failed_cases.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            failed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return report