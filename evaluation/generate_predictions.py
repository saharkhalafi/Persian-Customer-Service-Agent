from __future__ import annotations

import json
from pathlib import Path

from bert_score import score

from app.api.dependencies import get_agent, get_gemini_client
from app.core.context import RequestContext
from app.core.database import SessionLocal


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    "Evaluate_data/generation_eval_data.json"
)

RESULT_PATH = Path(
    "evaluation/results/generation_gemini_eval.json"
)

# فقط برای اجرای evaluation
# می‌توانی یک customer واقعی از دیتابیس قرار بدهی.
CUSTOMER_ID = "9206288"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "generation_eval_data.json must contain a JSON list."
        )

    return data


# ============================================================
# EXTRACT ANSWER
# ============================================================

def extract_answer(result) -> str:
    """
    Extract answer from AgentResult.
    """

    answer = getattr(
        result,
        "answer",
        None,
    )

    if answer is None:
        return ""

    return str(answer).strip()


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_dataset()

    print("=" * 70)
    print("GENERATION EVALUATION")
    print("=" * 70)

    print(f"Dataset : {DATASET_PATH}")
    print(f"Samples : {len(dataset)}")
    print(f"Customer: {CUSTOMER_ID}")
    print()

    # --------------------------------------------------------
    # Build DB + Agent
    # --------------------------------------------------------

    db = SessionLocal()

    try:

        gemini = get_gemini_client()

        agent = get_agent(
            db=db,
            gemini=gemini,
        )

        # ----------------------------------------------------
        # Request Context
        # ----------------------------------------------------

        context = RequestContext(
            customer_id=CUSTOMER_ID
        )

        results = []

        # ====================================================
        # GENERATE ANSWERS
        # ====================================================

        for index, item in enumerate(
            dataset,
            start=1,
        ):

            question = item["question"]

            ground_truth = item[
                "ground_truth"
            ]

            print(
                f"[{index}/{len(dataset)}] "
                f"{question}"
            )

            try:

                result = agent.run(
                    question,
                    context,
                )

                prediction = extract_answer(
                    result
                )

                # --------------------------------------------
                # Tool calls
                # --------------------------------------------

                tool_calls = []

                for call in getattr(
                    result,
                    "tool_calls",
                    [],
                ):

                    tool_name = getattr(
                        call,
                        "name",
                        str(call),
                    )

                    tool_calls.append(
                        tool_name
                    )

                print(
                    "Prediction:",
                    prediction
                )

                print(
                    "Tools:",
                    tool_calls
                )

                results.append(
                    {
                        "index": index,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "tool_calls": tool_calls,
                    }
                )

            except Exception as exc:

                print(
                    "ERROR:",
                    repr(exc)
                )

                results.append(
                    {
                        "index": index,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": "",
                        "tool_calls": [],
                        "error": repr(exc),
                    }
                )

            print()

        # ====================================================
        # CHECK EMPTY PREDICTIONS
        # ====================================================

        valid_results = [
            item
            for item in results
            if item["prediction"].strip()
        ]

        empty_results = [
            item
            for item in results
            if not item["prediction"].strip()
        ]

        print("=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)

        print(
            f"Total       : {len(results)}"
        )

        print(
            f"Valid       : {len(valid_results)}"
        )

        print(
            f"Empty       : {len(empty_results)}"
        )

        if empty_results:

            print("\nEmpty predictions:")

            for item in empty_results:

                print(
                    f"- {item['index']}: "
                    f"{item['question']}"
                )

        # ====================================================
        # STOP IF EVERYTHING FAILED
        # ====================================================

        if not valid_results:

            raise RuntimeError(
                "All predictions are empty. "
                "BERTScore cannot be calculated."
            )

        # ====================================================
        # BERTSCORE
        # ====================================================

        predictions = [
            item["prediction"]
            for item in valid_results
        ]

        references = [
            item["ground_truth"]
            for item in valid_results
        ]

        print()
        print("=" * 70)
        print("CALCULATING BERTSCORE")
        print("=" * 70)

        P, R, F1 = score(
            predictions,
            references,
            lang="fa",
            device="cuda",
            verbose=True,
            rescale_with_baseline=False,
        )

        # ====================================================
        # ADD PER-SAMPLE METRICS
        # ====================================================

        for i, item in enumerate(
            valid_results
        ):

            item[
                "bert_score_precision"
            ] = P[i].item()

            item[
                "bert_score_recall"
            ] = R[i].item()

            item[
                "bert_score_f1"
            ] = F1[i].item()

        # ====================================================
        # AVERAGES
        # ====================================================

        average_precision = (
            P.mean().item()
        )

        average_recall = (
            R.mean().item()
        )

        average_f1 = (
            F1.mean().item()
        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        output = {
            "metric": "BERTScore",

            "language": "fa",

            "dataset": str(
                DATASET_PATH
            ),

            "customer_id": CUSTOMER_ID,

            "num_samples": len(results),

            "valid_samples": len(
                valid_results
            ),

            "empty_samples": len(
                empty_results
            ),

            "average_precision": (
                average_precision
            ),

            "average_recall": (
                average_recall
            ),

            "average_f1": (
                average_f1
            ),

            "average_f1_percent": (
                average_f1 * 100
            ),

            "results": results,
        }

        # ====================================================
        # SAVE
        # ====================================================

        RESULT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with RESULT_PATH.open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                output,
                f,
                ensure_ascii=False,
                indent=2,
            )

        # ====================================================
        # REPORT
        # ====================================================

        print()
        print("=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        print(
            f"BERTScore Precision : "
            f"{average_precision:.4f}"
        )

        print(
            f"BERTScore Recall    : "
            f"{average_recall:.4f}"
        )

        print(
            f"BERTScore F1        : "
            f"{average_f1:.4f}"
        )

        print(
            f"BERTScore F1 (%)    : "
            f"{average_f1 * 100:.2f}%"
        )

        print("=" * 70)

        print()
        print(
            "Results saved to:"
        )

        print(
            RESULT_PATH
        )

    finally:

        db.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()