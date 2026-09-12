"""Rebuild aligned Colab retrieval artifacts from current sources.

Does not modify production RAG code or Evaluate_data/retrieval_eval_data.xlsx.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from app.core.gemini import GeminiClient
from app.retrieval.knowledge_retriever import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FAQ_PATH,
    _chunk_faq,
)

ARTIFACT_DIR = PROJECT_ROOT / "evaluation" / "colab_artifacts"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
XLSX_SOURCE = PROJECT_ROOT / "Evaluate_data" / "retrieval_eval_data.xlsx"
XLSX_DEST = ARTIFACT_DIR / "retrieval_eval_data.xlsx"


def parse_ids(raw) -> list[int]:
    text = "" if raw is None or (isinstance(raw, float) and np.isnan(raw)) else str(raw).strip()
    ids: list[int] = []
    for part in re.split(r"[,;\s]+", text):
        if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
            ids.append(int(part))
    return ids


def load_eval_rows(path: Path) -> list[dict]:
    frame = pd.read_excel(path)
    rows = []
    for index, row in frame.iterrows():
        question = str(row["question"]).strip()
        ids = parse_ids(row.get("relevant_chunk_ids"))
        item = {
            "row_index": int(index),
            "question": question,
            "relevant_chunk_ids": ids,
        }
        if "id" in frame.columns and pd.notna(row.get("id")):
            item["id"] = row["id"]
        rows.append(item)
    return rows


def embed_texts(gemini: GeminiClient, texts: list[str], task_type: str) -> np.ndarray:
    embeddings = gemini.embed(texts, task_type=task_type)
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch for {task_type}: "
            f"{len(embeddings)} vs {len(texts)}"
        )
    array = np.asarray(embeddings, dtype=np.float32)
    expected = (len(texts), GeminiClient.EMBEDDING_DIM)
    if array.shape != expected:
        raise RuntimeError(f"Unexpected embedding shape {array.shape}, expected {expected}")
    return array


def cosine_topk(query_vec: np.ndarray, chunk_matrix: np.ndarray, k: int) -> list[int]:
    scores = chunk_matrix @ query_vec
    k = min(k, chunk_matrix.shape[0])
    return np.argsort(-scores)[:k].tolist()


def build_verification(
    queries: list[dict],
    chunks: list[dict],
    query_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    xlsx_path: Path,
) -> dict:
    chunk_ids = [int(chunk["chunk_id"]) for chunk in chunks]
    chunk_id_set = set(chunk_ids)
    by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}

    excel_count = len(pd.read_excel(xlsx_path))
    checks = {
        "query_count_matches_excel": len(queries) == excel_count,
        "query_embedding_count_matches_queries": query_embeddings.shape[0] == len(queries),
        "chunk_embedding_count_matches_chunks_json": chunk_embeddings.shape[0] == len(chunks),
        "chunk_ids_unique": len(chunk_ids) == len(set(chunk_ids)),
        "chunk_ids_are_0_to_n_minus_1": chunk_ids == list(range(len(chunks))),
        "all_relevant_chunk_ids_exist": True,
    }

    per_query = []
    missing_ids: list[dict] = []
    dense_hits_at_10 = 0

    for index, query in enumerate(queries):
        relevant = query["relevant_chunk_ids"]
        absent = [chunk_id for chunk_id in relevant if chunk_id not in chunk_id_set]
        if absent:
            checks["all_relevant_chunk_ids_exist"] = False
            missing_ids.append({"row_index": index, "missing_ids": absent})

        relevant_chunks = []
        for chunk_id in relevant:
            chunk = by_id.get(chunk_id)
            relevant_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": None if chunk is None else chunk.get("page"),
                    "content": None if chunk is None else chunk["content"],
                }
            )

        top10 = cosine_topk(query_embeddings[index], chunk_embeddings, 10)
        hit_at_10 = any(chunk_id in set(relevant) for chunk_id in top10)
        if hit_at_10:
            dense_hits_at_10 += 1

        per_query.append(
            {
                "row_index": index,
                "question": query["question"],
                "relevant_chunk_ids": relevant,
                "relevant_chunks": relevant_chunks,
                "dense_top10_chunk_ids": top10,
                "dense_hit_at_10": hit_at_10,
            }
        )

    n = len(queries) or 1
    checks["dense_recall_at_10"] = round(dense_hits_at_10 / n, 4)

    return {
        "passed": all(
            value is True
            for key, value in checks.items()
            if key != "dense_recall_at_10"
        ),
        "checks": checks,
        "missing_relevant_ids": missing_ids,
        "shapes": {
            "excel_rows": excel_count,
            "queries": len(queries),
            "query_embeddings": list(query_embeddings.shape),
            "chunks": len(chunks),
            "chunk_embeddings": list(chunk_embeddings.shape),
        },
        "per_query": per_query,
    }


def main() -> None:
    if not XLSX_SOURCE.is_file():
        raise FileNotFoundError(XLSX_SOURCE)
    if not FAQ_PATH.is_file():
        raise FileNotFoundError(FAQ_PATH)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(XLSX_SOURCE, XLSX_DEST)
    queries = load_eval_rows(XLSX_DEST)
    chunks = [
        {
            "chunk_id": int(chunk["chunk_id"]),
            "page": int(chunk.get("page") or 0),
            "content": chunk["content"],
        }
        for chunk in _chunk_faq(FAQ_PATH)
    ]
    if not chunks:
        raise RuntimeError("Production chunking returned no chunks.")
    if not queries:
        raise RuntimeError("No retrieval evaluation queries found.")

    gemini = GeminiClient()
    chunk_embeddings = embed_texts(
        gemini,
        [chunk["content"] for chunk in chunks],
        "RETRIEVAL_DOCUMENT",
    )
    query_embeddings = embed_texts(
        gemini,
        [query["question"] for query in queries],
        "RETRIEVAL_QUERY",
    )

    chunks_path = ARTIFACT_DIR / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(ARTIFACT_DIR / "chunk_embeddings.npy", chunk_embeddings)
    np.save(ARTIFACT_DIR / "query_embeddings.npy", query_embeddings)
    np.save(RESULTS_DIR / "chunk_embeddings.npy", chunk_embeddings)
    np.save(RESULTS_DIR / "query_embeddings.npy", query_embeddings)

    metadata = {
        "number_of_queries": len(queries),
        "number_of_chunks": len(chunks),
        "embedding_model": GeminiClient.EMBEDDING_MODEL,
        "embedding_dim": GeminiClient.EMBEDDING_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "creation_source": {
            "script": "evaluation/build_colab_retrieval_artifacts.py",
            "faq_pdf": str(FAQ_PATH),
            "chunking": "app.retrieval.knowledge_retriever._chunk_faq",
            "queries_and_ground_truth": str(XLSX_SOURCE),
            "query_task_type": "RETRIEVAL_QUERY",
            "chunk_task_type": "RETRIEVAL_DOCUMENT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "query_order": [query["question"] for query in queries],
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
        "files": {
            "queries_ground_truth": "retrieval_eval_data.xlsx",
            "chunks": "chunks.json",
            "query_embeddings": "query_embeddings.npy",
            "chunk_embeddings": "chunk_embeddings.npy",
            "verification": "verification_report.json",
        },
        "query_embeddings_shape": list(query_embeddings.shape),
        "chunk_embeddings_shape": list(chunk_embeddings.shape),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    verification = build_verification(
        queries,
        chunks,
        query_embeddings,
        chunk_embeddings,
        XLSX_DEST,
    )
    (ARTIFACT_DIR / "verification_report.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = """# Colab retrieval artifacts

Load files in this order:

1. `retrieval_eval_data.xlsx` — queries and `relevant_chunk_ids` (source of truth for Ground Truth).
2. `chunks.json` — production FAQ chunks, IDs `0 .. N-1`, same order as embeddings.
3. `chunk_embeddings.npy` — row `i` is the embedding of `chunks.json[i]`.
4. `query_embeddings.npy` — row `i` is the embedding of Excel row `i` (`query_order` in `metadata.json`).
5. `metadata.json` — counts, model, chunk size/overlap, query order, chunk IDs.
6. `verification_report.json` — alignment checks; inspect `per_query` for question + relevant chunk text.

Source of truth:

- Ground Truth / query text / query order: `retrieval_eval_data.xlsx`
- Chunk text / chunk IDs / chunk order: `chunks.json`
- Do not pair these embeddings with `generation_eval_data.json` (different questions).
"""
    (ARTIFACT_DIR / "COLAB_README.md").write_text(readme, encoding="utf-8")

    print("queries", len(queries))
    print("chunks", len(chunks))
    print("query_embeddings", query_embeddings.shape)
    print("chunk_embeddings", chunk_embeddings.shape)
    print("verification_passed", verification["passed"])
    print("dense_recall_at_10", verification["checks"]["dense_recall_at_10"])
    print("written", ARTIFACT_DIR)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()
