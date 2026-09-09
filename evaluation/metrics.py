from collections import Counter
from typing import Any


def normalize_tools(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        if value.lower() == "none":
            return []
        return [value]

    if isinstance(value, list):
        return value

    return []


def tool_selection_metrics(results: list[dict]) -> dict:
    tp = 0
    fp = 0
    fn = 0
    exact = 0

    for r in results:
        expected = normalize_tools(r.get("expected_tool"))
        actual = [
            tool["name"]
            for tool in r.get("actual_tools", [])
        ]

        expected_set = set(expected)
        actual_set = set(actual)

        if expected_set == actual_set:
            exact += 1

        tp += len(expected_set & actual_set)
        fp += len(actual_set - expected_set)
        fn += len(expected_set - actual_set)

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return {
        "tool_exact_match": exact / len(results) if results else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": len(results),
    }


def argument_exact_match(results: list[dict]) -> dict:
    evaluated = 0
    correct = 0

    for r in results:
        expected = r.get("expected_arguments")

        if expected is None:
            continue

        evaluated += 1

        actual = [
            tool
            for tool in r.get("actual_tools", [])
        ]

        if isinstance(expected, dict):

            if len(actual) != 1:
                continue

            if actual[0]["arguments"] == expected:
                correct += 1

        elif isinstance(expected, list):

            if len(actual) != len(expected):
                continue

            matched = True

            for actual_call, expected_call in zip(
                actual,
                expected,
            ):
                if not arguments_match(
                    actual_call["arguments"],
                    expected_call,
                ):
                    matched = False
                    break

            if matched:
                correct += 1

    return {
        "exact_match": correct / evaluated if evaluated else 0,
        "correct": correct,
        "evaluated": evaluated,
    }


def arguments_match(
    actual: dict,
    expected: dict,
) -> bool:

    for key, expected_value in expected.items():

        if isinstance(expected_value, str):
            if expected_value.startswith("<resolved_from_step_"):
                if key not in actual:
                    return False
                continue

        if actual.get(key) != expected_value:
            return False

    return True


def confusion_matrix(results: list[dict]) -> dict:
    matrix = Counter()

    for r in results:
        expected = r.get("expected_intent")

        tools = r.get("actual_tools", [])

        if not tools:
            actual = "NONE"
        else:
            actual = tools[0]["name"]

        matrix[(expected, actual)] += 1

    return {
        f"{expected} -> {actual}": count
        for (expected, actual), count in matrix.items()
    }


def guardrail_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = 0

    for r in results:
        expected = r.get("expected_behavior")

        tools = r.get("actual_tools", [])
        answer = r.get("actual_answer", "").lower()

        no_tools = len(tools) == 0

        if expected == "OUT_OF_SCOPE":
            if no_tools:
                correct += 1

        elif expected == "ABSTAIN":
            if no_tools:
                correct += 1

        elif expected in {
            "DENY_UNAUTHORIZED",
            "BLOCK",
        }:
            if no_tools:
                correct += 1

    return {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
    }


def latency_metrics(results: list[dict]) -> dict:
    values = sorted(
        r["latency_ms"]
        for r in results
        if r.get("latency_ms") is not None
    )

    if not values:
        return {}

    def percentile(p: float):
        index = int((len(values) - 1) * p)
        return values[index]

    return {
        "mean_ms": round(sum(values) / len(values), 2),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "min_ms": values[0],
        "max_ms": values[-1],
    }