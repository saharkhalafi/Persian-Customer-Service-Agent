"""Generate agent answers for FAQ eval questions and score valid ones with BERTScore.

Usage (from repo root):
    python -m evaluation.generation_gemini_eval
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from bert_score import score as bert_score
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.api.dependencies import get_agent, get_gemini_client
from app.core.context import RequestContext
from app.core.database import SessionLocal


DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "generation_eval_data.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
REPORT_PATH = RESULTS_DIR / "generation_gemini_eval.json"
PREDICTIONS_PATH = RESULTS_DIR / "generation_predictions.json"
ERRORS_PATH = RESULTS_DIR / "generation_errors.json"

CUSTOMER_ID = "9206288"
BERT_MODEL = "bert-base-multilingual-cased"
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


def extract_answer(result) -> str:
    if isinstance(result, str):
        answer = result.strip()
    else:
        raw = getattr(result, "answer", None)
        if raw is None:
            raise ValueError("Agent result does not contain 'answer'.")
        answer = str(raw).strip()

    if not answer:
        raise ValueError("Agent returned an empty answer.")

    return answer


def serialize_tool_calls(result) -> list[dict]:
    records = []

    for call in getattr(result, "tool_calls", None) or []:
        if hasattr(call, "name"):
            records.append(
                {
                    "name": call.name,
                    "arguments": dict(getattr(call, "arguments", {}) or {}),
                }
            )
        elif isinstance(call, dict):
            records.append(call)
        else:
            records.append({"name": str(call)})

    return records


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


def bert_score_kwargs(device: str) -> dict:
    kwargs = {
        "model_type": BERT_MODEL,
        "device": device,
        "verbose": True,
        "rescale_with_baseline": False,
    }

    try:
        from bert_score.utils import model2layers

        layers = model2layers.get(BERT_MODEL)
        if layers is not None:
            kwargs["num_layers"] = layers
    except Exception:
        pass

    return kwargs


def main() -> None:
    dataset = load_dataset()

    print("=" * 70)
    print("GENERATION EVALUATION")
    print("=" * 70)
    print(f"Dataset : {DATASET_PATH}")
    print(f"Samples : {len(dataset)}")
    print(f"Customer: {CUSTOMER_ID}")
    print()

    db = SessionLocal()

    try:
        gemini = get_gemini_client()
        agent = get_agent(db=db, gemini=gemini)
        context = RequestContext(customer_id=CUSTOMER_ID)

        predictions: list[dict] = []
        errors: list[dict] = []

        for index, item in enumerate(dataset, start=1):
            question = item["question"]
            ground_truth = item["ground_truth"]
            print(f"[{index}/{len(dataset)}] {question}")

            row = {
                "index": index,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": None,
                "tool_calls": [],
                "error": None,
            }

            try:
                result = run_agent(agent, context, question)
                row["prediction"] = extract_answer(result)
                row["tool_calls"] = serialize_tool_calls(result)
                predictions.append(row)
                print(f"Prediction: {row['prediction']}")
                print(f"Tools: {[call.get('name') for call in row['tool_calls']]}")
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(row)
                print(f"ERROR: {row['error']}")

            print()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        with PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

        with ERRORS_PATH.open("w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

        print("=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        print(f"Total        : {len(dataset)}")
        print(f"Successful   : {len(predictions)}")
        print(f"Errors       : {len(errors)}")
        print(f"Predictions  : {PREDICTIONS_PATH}")
        print(f"Errors file  : {ERRORS_PATH}")

        if errors:
            print("\nFailed samples:")
            for row in errors:
                print(f"- {row['index']}: {row['question']}")
                print(f"  {row['error']}")

        bert_metrics = None
        scored_rows = list(predictions)

        if predictions:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print()
            print("=" * 70)
            print("CALCULATING BERTSCORE")
            print("=" * 70)
            print(f"Model   : {BERT_MODEL}")
            print(f"Device  : {device}")
            print(f"Samples : {len(predictions)}")
            print(
                "Note    : lang='fa' is not in this bert-score build "
                "(only en/zh/tr/en-sci); using multilingual BERT."
            )

            P, R, F1 = bert_score(
                [row["prediction"] for row in predictions],
                [row["ground_truth"] for row in predictions],
                **bert_score_kwargs(device),
            )

            for i, row in enumerate(scored_rows):
                row["bert_score_precision"] = P[i].item()
                row["bert_score_recall"] = R[i].item()
                row["bert_score_f1"] = F1[i].item()

            bert_metrics = {
                "model_type": BERT_MODEL,
                "device": device,
                "num_samples_evaluated": len(predictions),
                "average_precision": P.mean().item(),
                "average_recall": R.mean().item(),
                "average_f1": F1.mean().item(),
                "average_f1_percent": F1.mean().item() * 100,
            }

            print()
            print("=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(f"BERTScore samples   : {bert_metrics['num_samples_evaluated']}")
            print(f"BERTScore Precision : {bert_metrics['average_precision']:.4f}")
            print(f"BERTScore Recall    : {bert_metrics['average_recall']:.4f}")
            print(f"BERTScore F1        : {bert_metrics['average_f1']:.4f}")
            print(f"BERTScore F1 (%)    : {bert_metrics['average_f1_percent']:.2f}%")
        else:
            print()
            print("BERTScore skipped: no valid predictions.")

        report = {
            "metric": "BERTScore" if bert_metrics else None,
            "language": "fa",
            "bert_model_type": BERT_MODEL,
            "dataset": str(DATASET_PATH),
            "customer_id": CUSTOMER_ID,
            "num_samples": len(dataset),
            "successful_samples": len(predictions),
            "error_samples": len(errors),
            "bertscore": bert_metrics,
            "predictions": scored_rows,
            "errors": errors,
        }

        with REPORT_PATH.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print()
        print(f"Report saved to: {REPORT_PATH}")
        print("=" * 70)

        if len(predictions) + len(errors) != len(dataset):
            raise RuntimeError("Not all dataset samples were processed.")

    finally:
        db.close()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    main()
