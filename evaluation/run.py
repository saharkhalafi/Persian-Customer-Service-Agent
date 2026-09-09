import argparse
import json

from evaluation.runner import EvaluationRunner
from evaluation.report import save_report


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--customer-id",
        required=True,
    )

    args = parser.parse_args()

    runner = EvaluationRunner(
        dataset_path=args.dataset,
        customer_id=args.customer_id,
    )

    results = runner.run()

    report = save_report(results)

    print("\n" + "=" * 60)
    print("EVALUATION RESULT")
    print("=" * 60)

    print(
        "Total:",
        report["summary"]["total_cases"],
    )

    print(
        "Errors:",
        report["summary"]["failed_cases"],
    )

    print("\nIntent Router:")
    print(
        report["intent_router"]["tool_metrics"]
    )

    print("\nTool Calling:")
    print(
        report["tool_calling"]["tool_metrics"]
    )

    print("\nArguments:")
    print(
        report["tool_calling"]["argument_metrics"]
    )

    print("\nGuardrails:")
    print(
        report["guardrails"]
    )

    print("\nLatency:")
    print(
        report["latency"]
    )


if __name__ == "__main__":
    main()