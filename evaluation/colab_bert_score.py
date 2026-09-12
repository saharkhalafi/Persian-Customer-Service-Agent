"""Offline Persian BERTScore for Colab (no Gemini / PostgreSQL / FastAPI).

Usage:
    python -m evaluation.colab_bert_score
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from bert_score import score as bert_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "evaluation" / "colab_artifacts"
PREDICTIONS_PATH = ARTIFACT_DIR / "generation_predictions.json"
OUTPUT_PATH = ARTIFACT_DIR / "bert_score_results.json"
BERT_MODEL = "bert-base-multilingual-cased"


def load_predictions() -> list[dict]:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PREDICTIONS_PATH}. "
            "Run generation_gemini_eval and prepare_colab_eval first."
        )

    with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("generation_predictions.json must be a JSON list.")

    return data


def valid_rows(rows: list[dict]) -> list[dict]:
    selected = []

    for row in rows:
        if row.get("error"):
            continue
        prediction = str(row.get("prediction") or "").strip()
        reference = str(row.get("ground_truth") or "").strip()
        if not prediction or not reference:
            continue
        selected.append(row)

    return selected


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
    rows = load_predictions()
    scored = valid_rows(rows)

    print("=" * 70)
    print("COLAB BERTSCORE")
    print("=" * 70)
    print(f"File     : {PREDICTIONS_PATH}")
    print(f"Loaded   : {len(rows)}")
    print(f"Valid    : {len(scored)}")

    if not scored:
        raise RuntimeError("No valid predictions to score.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device   : {device}")
    print(f"Model    : {BERT_MODEL}")
    print(
        "lang='fa' is not supported by this bert-score install; "
        "using bert-base-multilingual-cased."
    )

    P, R, F1 = bert_score(
        [row["prediction"] for row in scored],
        [row["ground_truth"] for row in scored],
        **bert_score_kwargs(device),
    )

    details = []

    for i, row in enumerate(scored):
        details.append(
            {
                "index": row.get("index", i + 1),
                "question": row.get("question"),
                "ground_truth": row.get("ground_truth"),
                "prediction": row.get("prediction"),
                "precision": P[i].item(),
                "recall": R[i].item(),
                "f1": F1[i].item(),
                "f1_percent": F1[i].item() * 100,
            }
        )

    output = {
        "metric": "BERTScore",
        "language": "fa",
        "model_type": BERT_MODEL,
        "device": device,
        "num_samples_loaded": len(rows),
        "num_samples_evaluated": len(scored),
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "f1_percent": F1.mean().item() * 100,
        "results": details,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print(f"Precision : {output['precision']:.4f}")
    print(f"Recall    : {output['recall']:.4f}")
    print(f"F1        : {output['f1']:.4f}")
    print(f"F1%       : {output['f1_percent']:.2f}%")
    print(f"Evaluated : {output['num_samples_evaluated']}")
    print(f"Saved     : {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    main()
