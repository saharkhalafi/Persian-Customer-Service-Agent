"""Isolated Gemini FAQ retrieval evaluation.

Re-runs retrieval metrics for generation_eval_data.json using the new
Gemini embedding + BM25 pipeline. Does not modify production indexes
unless they are empty (KnowledgeRetriever is not used for writes).

Usage (from repo root):
    .venv/Scripts/python.exe -m evaluation.retrieval_gemini_eval
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from chromadb import PersistentClient
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from app.core.gemini import GeminiClient
from app.retrieval.knowledge_retriever import (
    COLLECTION_NAME,
    FAQ_PATH,
    INITIAL_K,
    _bm25_rerank,
    _chunk_faq,
)


DATASET_PATH = PROJECT_ROOT / "Evaluate_data" / "generation_eval_data.json"
XLSX_PATH = PROJECT_ROOT / "Evaluate_data" / "retrieval_eval_data.xlsx"
OLD_PREVIEW_PATH = PROJECT_ROOT / "data" / "chunks_preview.txt"
EVAL_CHROMA_DIR = PROJECT_ROOT / "Evaluate_data" / "chroma_faq_gemini_eval"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
FINAL_K = 3

# Human-style relevance on the PREVIOUS chunk_ids (chunks_preview.txt).
# generation_eval_data.json stores the same retrieved_chunks for every query
# (dummy-embedding artifact), so those IDs are not used as labels.
# Each ID is the old chunk whose content actually answers the question.
OLD_RELEVANT_BY_QUESTION: dict[str, list[int]] = {
    "مهلت مرجوعی کالا در مدیسه چقدر است؟": [9, 10],
    "هزینه پست مرجوعی را کی پرداخت می‌کند؟": [9],
    "برای مرجوع کردن کالا باید چه شرایطی داشته باشد؟": [10],
    "اگر کالا را نپسندیدم می‌توانم مرجوع کنم؟": [10],
    "بعد از مرجوعی پول چطور برمی‌گردد؟": [11],
    "کد تخفیف بعد از مرجوعی دوباره فعال می‌شود؟": [13],
    "ثبت مرجوعی از چه طریقی ممکن است؟": [8, 9],
    "آیا خرید تلفنی یا حضوری امکان‌پذیر است؟": [1],
    "چطور رمز عبور را بازیابی کنم؟": [0],
    "پرداخت در محل برای کدام شهرها ممکن است؟": [4],
    "چطور سایز مناسب را انتخاب کنم؟": [5],
    "امکان پرداخت اقساطی وجود دارد؟": [6],
    "کد رهگیری مرسوله را از کجا بگیرم؟": [6, 7],
    "اگر تأخیر در ارسال باشد مدیسه اطلاع می‌دهد؟": [8],
    "کد تخفیف اولین خرید چیست؟": [12],
    "اگر کارت هدیه مفقود شود چه می‌شود؟": [15],
    "چطور مدیسو کسب کنم؟": [16],
    "تاریخ انقضای لوازم بهداشتی اعلام می‌شود؟": [11],
    "چطور خبرنامه را لغو کنم؟": [1],
    "آیا می‌توانم سفارش را به شخص دیگری تحویل دهم؟": [3, 4],
}


def normalize_text(text: str) -> str:
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_old_preview(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    parts = re.split(r"\[(\s*\d+)\]\s", raw)
    chunks = []

    for index in range(1, len(parts), 2):
        chunk_id = int(parts[index].strip())
        content = parts[index + 1].strip()
        chunks.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "normalized": normalize_text(content),
            }
        )

    return chunks


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[0-9A-Za-z\u0600-\u06FF]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)

    if not left or not right:
        return 0.0

    return len(left & right) / len(left | right)


def map_old_to_new(
    old_chunks: list[dict],
    new_chunks: list[dict],
) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}

    for old in old_chunks:
        exact = [
            new["chunk_id"]
            for new in new_chunks
            if new["normalized"] == old["normalized"]
        ]

        if exact:
            mapping[old["chunk_id"]] = exact
            continue

        scored = []

        for new in new_chunks:
            score = jaccard(old["normalized"], new["normalized"])
            contained = (
                old["normalized"] in new["normalized"]
                or new["normalized"] in old["normalized"]
            )

            if contained:
                score = max(score, 0.99)

            scored.append((score, new["chunk_id"]))

        scored.sort(reverse=True)
        best_score, best_id = scored[0]
        mapping[old["chunk_id"]] = [best_id] if best_score >= 0.45 else []

    return mapping


def remap_ids(old_ids: list[int], mapping: dict[int, list[int]]) -> list[int]:
    new_ids: list[int] = []

    for old_id in old_ids:
        for new_id in mapping.get(old_id, []):
            if new_id not in new_ids:
                new_ids.append(new_id)

    return new_ids


def chunks_identical(
    old_chunks: list[dict],
    new_chunks: list[dict],
) -> bool:
    if len(old_chunks) != len(new_chunks):
        return False

    for old, new in zip(old_chunks, new_chunks):
        if old["chunk_id"] != new["chunk_id"]:
            return False

        if old["normalized"] != new["normalized"]:
            return False

    return True


def ensure_eval_index(
    gemini: GeminiClient,
    new_chunks: list[dict],
) -> PersistentClient:
    if EVAL_CHROMA_DIR.exists():
        shutil.rmtree(EVAL_CHROMA_DIR)

    EVAL_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(EVAL_CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embeddings = gemini.embed(
        [chunk["content"] for chunk in new_chunks],
        task_type="RETRIEVAL_DOCUMENT",
    )

    if len(embeddings) != len(new_chunks):
        raise RuntimeError(
            f"Embedding count mismatch: {len(embeddings)} vs {len(new_chunks)}"
        )

    collection.add(
        ids=[str(chunk["chunk_id"]) for chunk in new_chunks],
        documents=[chunk["content"] for chunk in new_chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk.get("page", 0),
            }
            for chunk in new_chunks
        ],
    )

    return client


def retrieve_top3(
    gemini: GeminiClient,
    collection,
    query: str,
) -> list[dict]:
    count = collection.count()
    query_embedding = gemini.embed(query, task_type="RETRIEVAL_QUERY")

    if not query_embedding or count == 0:
        return []

    initial_k = min(max(FINAL_K, INITIAL_K), count)
    raw = collection.query(
        query_embeddings=query_embedding,
        n_results=initial_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results = []

    for index, content in enumerate(documents):
        if not content:
            continue

        metadata = metadatas[index] if index < len(metadatas) else {} or {}
        distance = distances[index] if index < len(distances) else None
        chunk_id = metadata.get("chunk_id")

        try:
            chunk_id = int(chunk_id)
        except (TypeError, ValueError):
            chunk_id = index

        results.append(
            {
                "chunk_id": chunk_id,
                "content": str(content).strip(),
                "score": float(1 - distance) if distance is not None else None,
            }
        )

    return _bm25_rerank(query, results)[:FINAL_K]


def metrics_for_rows(rows: list[dict]) -> dict[str, float]:
    hits = []
    mrrs = []
    precs = []
    recs = []

    for row in rows:
        relevant = set(row["relevant_chunk_ids"])
        retrieved = row["retrieved_top3"]
        overlap = [chunk_id for chunk_id in retrieved if chunk_id in relevant]
        hit_count = len(overlap)
        hits.append(1 if hit_count else 0)

        rank = 0.0

        for position, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                rank = 1 / position
                break

        mrrs.append(rank)
        precs.append(hit_count / len(retrieved) if retrieved else 0.0)
        recs.append(hit_count / len(relevant) if relevant else 0.0)

    n = len(rows)

    return {
        "Hit Rate @3": round(sum(hits) / n, 3),
        "MRR": round(sum(mrrs) / n, 3),
        "Precision @3": round(sum(precs) / n, 3),
        "Recall @3": round(sum(recs) / n, 3),
    }


def load_xlsx_labels() -> list[dict]:
    import pandas as pd

    frame = pd.read_excel(XLSX_PATH)
    rows = []

    for _, row in frame.iterrows():
        raw_ids = str(row.get("relevant_chunk_ids", "")).strip()
        ids = []

        for part in re.split(r"[,;\s]+", raw_ids):
            if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                ids.append(int(part))

        rows.append(
            {
                "question": str(row["question"]).strip(),
                "old_relevant": ids,
            }
        )

    return rows


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    if not OLD_PREVIEW_PATH.is_file():
        raise FileNotFoundError(OLD_PREVIEW_PATH)

    if not FAQ_PATH.is_file():
        raise FileNotFoundError(FAQ_PATH)

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    old_chunks = parse_old_preview(OLD_PREVIEW_PATH)
    new_raw = _chunk_faq(FAQ_PATH)
    new_chunks = [
        {
            "chunk_id": chunk["chunk_id"],
            "page": chunk.get("page", 0),
            "content": chunk["content"],
            "normalized": normalize_text(chunk["content"]),
        }
        for chunk in new_raw
    ]

    identical = chunks_identical(old_chunks, new_chunks)
    mapping = map_old_to_new(old_chunks, new_chunks)
    unmapped = [
        old_id
        for old_id, new_ids in mapping.items()
        if not new_ids
    ]

    print("=" * 72)
    print("CHUNK COMPARISON")
    print("=" * 72)
    print(f"Old chunks (preview): {len(old_chunks)}")
    print(f"New chunks (pymupdf 600/100): {len(new_chunks)}")
    print(f"IDs and content identical: {identical}")
    print(f"Unmapped old IDs: {unmapped or 'none'}")

    if identical:
        print("Reusing previous relevant chunk_ids without remapping.")
    else:
        print("Chunking changed. Remapping relevance labels by content.")
        for old_id, new_ids in mapping.items():
            if new_ids != [old_id]:
                print(f"  old {old_id} -> new {new_ids}")

    gemini = GeminiClient()
    client = ensure_eval_index(gemini, new_chunks)
    collection = client.get_collection(COLLECTION_NAME)

    eval_rows = []
    missing_label_questions = []

    for item in dataset:
        question = item["question"].strip()
        old_ids = OLD_RELEVANT_BY_QUESTION.get(question)

        if not old_ids:
            missing_label_questions.append(question)
            continue

        relevant = remap_ids(old_ids, mapping) if not identical else list(old_ids)
        retrieved_docs = retrieve_top3(gemini, collection, question)
        retrieved_ids = [doc["chunk_id"] for doc in retrieved_docs]
        overlap = [chunk_id for chunk_id in retrieved_ids if chunk_id in set(relevant)]

        eval_rows.append(
            {
                "question": question,
                "ground_truth": item.get("ground_truth"),
                "old_relevant_chunk_ids": old_ids,
                "relevant_chunk_ids": relevant,
                "retrieved_top3": retrieved_ids,
                "hit": bool(overlap),
                "missed_relevant": [
                    chunk_id for chunk_id in relevant if chunk_id not in retrieved_ids
                ],
            }
        )

    if missing_label_questions:
        raise RuntimeError(
            "Missing relevance mapping for: " + " | ".join(missing_label_questions)
        )

    generation_metrics = metrics_for_rows(eval_rows)
    misses = [row for row in eval_rows if not row["hit"]]

    xlsx_metrics = None
    xlsx_rows = []

    if XLSX_PATH.is_file():
        for item in load_xlsx_labels():
            relevant = (
                remap_ids(item["old_relevant"], mapping)
                if not identical
                else list(item["old_relevant"])
            )
            retrieved_docs = retrieve_top3(gemini, collection, item["question"])
            retrieved_ids = [doc["chunk_id"] for doc in retrieved_docs]
            overlap = [chunk_id for chunk_id in retrieved_ids if chunk_id in set(relevant)]
            xlsx_rows.append(
                {
                    "question": item["question"],
                    "old_relevant_chunk_ids": item["old_relevant"],
                    "relevant_chunk_ids": relevant,
                    "retrieved_top3": retrieved_ids,
                    "hit": bool(overlap),
                }
            )

        xlsx_metrics = metrics_for_rows(xlsx_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "configuration": {
            "embedding_model": GeminiClient.EMBEDDING_MODEL,
            "reranker_model": "BM25 (hybrid 0.6 dense + 0.4 lexical on top-10)",
            "chunk_size": 600,
            "chunk_overlap": 100,
            "collection": COLLECTION_NAME,
            "initial_k": INITIAL_K,
            "final_k": FINAL_K,
            "knowledge_source": str(FAQ_PATH),
            "dataset": str(DATASET_PATH),
        },
        "chunk_alignment": {
            "old_chunk_count": len(old_chunks),
            "new_chunk_count": len(new_chunks),
            "identical_ids_and_content": identical,
            "old_to_new_mapping": {
                str(old_id): new_ids for old_id, new_ids in mapping.items()
            },
            "label_source": (
                "previous chunk_ids reused"
                if identical
                else "remapped by chunk content from previous labels"
            ),
            "generation_json_retrieved_chunks_reused": False,
            "generation_json_retrieved_chunks_note": (
                "Not used as labels: every query has the same IDs "
                "[2, 10, 14, 15, 17]."
            ),
        },
        "generation_eval_data": {
            "n_queries": len(eval_rows),
            "n_chunks": len(new_chunks),
            "metrics": generation_metrics,
            "misses": [
                {
                    "question": row["question"],
                    "relevant_chunk_ids": row["relevant_chunk_ids"],
                    "retrieved_top3": row["retrieved_top3"],
                }
                for row in misses
            ],
            "per_query": eval_rows,
        },
        "official_retrieval_xlsx": {
            "n_queries": len(xlsx_rows),
            "metrics": xlsx_metrics,
            "per_query": xlsx_rows,
        }
        if xlsx_metrics
        else None,
    }

    out_path = RESULTS_DIR / "retrieval_gemini_eval.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("GENERATION_EVAL_DATA.JSON  (n={})".format(len(eval_rows)))
    print("=" * 72)
    print(
        f"| Gemini knowledge pipeline | {GeminiClient.EMBEDDING_MODEL} | "
        f"BM25 hybrid | {generation_metrics['Hit Rate @3']} | "
        f"{generation_metrics['MRR']} | {generation_metrics['Precision @3']} | "
        f"{generation_metrics['Recall @3']} |"
    )
    print(f"Number of queries: {len(eval_rows)}")
    print(f"Number of chunks: {len(new_chunks)}")
    print()
    print("Top-3 retrieved chunk IDs per query:")

    for row in eval_rows:
        print(
            f"- {row['question']}\n"
            f"    relevant={row['relevant_chunk_ids']}  "
            f"top3={row['retrieved_top3']}  hit={row['hit']}"
        )

    print()

    if misses:
        print("Queries whose relevant chunks were not retrieved:")
        for row in misses:
            print(f"- {row['question']}  relevant={row['relevant_chunk_ids']}  top3={row['retrieved_top3']}")
    else:
        print("Queries whose relevant chunks were not retrieved: none")

    if xlsx_metrics:
        print()
        print("=" * 72)
        print("RETRIEVAL_EVAL_DATA.XLSX  (n={})".format(len(xlsx_rows)))
        print("=" * 72)
        print(
            f"| Official labeled set | {GeminiClient.EMBEDDING_MODEL} | "
            f"BM25 hybrid | {xlsx_metrics['Hit Rate @3']} | "
            f"{xlsx_metrics['MRR']} | {xlsx_metrics['Precision @3']} | "
            f"{xlsx_metrics['Recall @3']} |"
        )

    print()
    print(f"Saved report -> {out_path}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    main()
