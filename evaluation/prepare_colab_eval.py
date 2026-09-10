"""Copy a self-contained offline evaluation folder for Colab.

Usage (from repo root):
    python -m evaluation.prepare_colab_eval
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.core.gemini import GeminiClient
from app.retrieval.knowledge_retriever import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_DIR,
    COLLECTION_NAME,
    INITIAL_K,
)


ARTIFACT_DIR = PROJECT_ROOT / "evaluation" / "colab_artifacts"
DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "generation_eval_data.json"
RETRIEVAL_XLSX = PROJECT_ROOT / "Evaluate_data" / "retrieval_eval_data.xlsx"
CHUNKS_PREVIEW = PROJECT_ROOT / "data" / "chunks_preview.txt"
PREDICTIONS_CANDIDATES = [
    PROJECT_ROOT / "evaluation" / "results" / "generation_predictions.json",
    PROJECT_ROOT / "Evaluate_data" / "generation_predictions.json",
]
CHUNK_EMBEDDINGS = PROJECT_ROOT / "evaluation" / "results" / "chunk_embeddings.npy"
QUERY_EMBEDDINGS = PROJECT_ROOT / "evaluation" / "results" / "query_embeddings.npy"


def copy_file(src: Path, dest: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Required file not found: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def copy_predictions(dest: Path) -> Path:
    for src in PREDICTIONS_CANDIDATES:
        if src.exists():
            copy_file(src, dest)
            return src
    raise FileNotFoundError(
        "generation_predictions.json not found. "
        "Run `python -m evaluation.generation_gemini_eval` first."
    )


def copy_chroma(dest: Path) -> str | None:
    if dest.exists():
        shutil.rmtree(dest)

    if not CHROMA_DIR.exists() or not (CHROMA_DIR / "chroma.sqlite3").exists():
        print(f"Chroma collection not found at {CHROMA_DIR}; skipped.")
        return None

    shutil.copytree(CHROMA_DIR, dest)
    return str(CHROMA_DIR)


def git_version() -> str | None:
    head = PROJECT_ROOT / ".git" / "HEAD"
    if not head.exists():
        return None

    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref = PROJECT_ROOT / ".git" / value.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()[:12]
        return value
    return value[:12]


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    pred_src = copy_predictions(ARTIFACT_DIR / "generation_predictions.json")
    copy_file(DATASET_PATH, ARTIFACT_DIR / "generation_eval_data.json")
    copy_file(RETRIEVAL_XLSX, ARTIFACT_DIR / "retrieval_eval_data.xlsx")
    copy_file(CHUNKS_PREVIEW, ARTIFACT_DIR / "chunks_preview.txt")
    copy_file(CHUNK_EMBEDDINGS, ARTIFACT_DIR / "chunk_embeddings.npy")
    copy_file(QUERY_EMBEDDINGS, ARTIFACT_DIR / "query_embeddings.npy")
    chroma_src = copy_chroma(ARTIFACT_DIR / "chroma_faq_gemini")

    metadata = {
        "project": "RAG FAQ modiseh",
        "git_revision": git_version(),
        "generation_model": "gemini-2.5-flash",
        "embedding_model": GeminiClient.EMBEDDING_MODEL,
        "embedding_dim": GeminiClient.EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "initial_k": INITIAL_K,
        "top_k": 3,
        "final_k": 3,
        "reranking": {
            "method": "hybrid BM25",
            "dense_weight": 0.6,
            "lexical_weight": 0.4,
            "candidates": INITIAL_K,
        },
        "chroma_collection": COLLECTION_NAME,
        "chroma_source": chroma_src,
        "bert_score_model": "bert-base-multilingual-cased",
        "predictions_source": str(pred_src),
        "dataset_source": str(DATASET_PATH),
        "chunk_embeddings": "chunk_embeddings.npy",
        "query_embeddings": "query_embeddings.npy",
        "chunk_embeddings_shape": [18, 768],
        "query_embeddings_shape": [20, 768],
    }

    with (ARTIFACT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Colab artifacts written to: {ARTIFACT_DIR}")
    for path in sorted(ARTIFACT_DIR.iterdir()):
        print(f"- {path.name}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    main()
