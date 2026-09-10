from pathlib import Path
import json

import numpy as np

from app.core.gemini import GeminiClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "Evaluate_data" / "generation_eval_data.json"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "colab_artifacts" / "query_embeddings.npy"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "query_embeddings.npy"


def main():
    # Load evaluation questions
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]

    print(f"Loaded {len(questions)} questions.")

    # Gemini client
    gemini = GeminiClient()

    # Gemini Embedding 001
    embeddings = gemini.embed(
        questions,
        task_type="RETRIEVAL_QUERY",
    )

    if len(embeddings) != len(questions):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(embeddings)} embeddings for "
            f"{len(questions)} questions."
        )

    embeddings_array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print("Embedding shape:", embeddings_array.shape)

    # Expected:
    # (20, 768)
    if embeddings_array.shape != (len(questions), 768):
        raise RuntimeError(
            f"Unexpected embedding shape: {embeddings_array.shape}"
        )

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, embeddings_array)
    np.save(RESULTS_PATH, embeddings_array)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {embeddings_array.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()