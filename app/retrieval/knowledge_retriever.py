from __future__ import annotations

from pathlib import Path
import math
import re
from collections import Counter

from chromadb import PersistentClient

from app.core.cache import TtlLruCache
from app.core.gemini import GeminiClient

_SEARCH_CACHE = TtlLruCache(maxsize=256, ttl_seconds=3600)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAQ_PATH = PROJECT_ROOT / "data" / "FAQ.pdf"
CHROMA_DIR = PROJECT_ROOT / "local_chroma_db-2026" / "gemini_faq"
LEGACY_CHROMA_DIR = PROJECT_ROOT / "local_chroma_db-2026" / "local_chroma_db"
COLLECTION_NAME = "faq_gemini"
LEGACY_COLLECTION = "langchain"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
INITIAL_K = 10
MAX_LIMIT = 10
SEPARATORS = ["\n\n", "\n", "۔", ". ", "!", "؟", " ", ""]


class KnowledgeRetriever:
    def __init__(self, gemini: GeminiClient):
        self.gemini = gemini
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._ensure_index()

    def search(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict]:
        limit = min(max(limit, 1), MAX_LIMIT)
        cache_key = (query.strip()[:500], limit)
        cached = _SEARCH_CACHE.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]

        count = self.collection.count()

        if count == 0:
            return []

        query_embedding = self.gemini.embed(
            query,
            task_type="RETRIEVAL_QUERY",
        )

        if not query_embedding:
            return []

        initial_k = min(max(limit, INITIAL_K), count)
        raw = self.collection.query(
            query_embeddings=query_embedding,
            n_results=initial_k,
            include=["documents", "metadatas", "distances"],
        )
        results = _parse_query_result(raw)

        if not results:
            _SEARCH_CACHE.set(cache_key, [])
            return []

        ranked = _bm25_rerank(query, results)[:limit]
        _SEARCH_CACHE.set(cache_key, [dict(item) for item in ranked])
        return ranked

    def _ensure_index(self) -> None:
        if self.collection.count() > 0:
            return

        chunks = _load_source_chunks()

        if not chunks:
            return

        embeddings = self.gemini.embed(
            [chunk["content"] for chunk in chunks],
            task_type="RETRIEVAL_DOCUMENT",
        )

        if len(embeddings) != len(chunks):
            return

        self.collection.add(
            ids=[str(chunk["chunk_id"]) for chunk in chunks],
            documents=[chunk["content"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "chunk_id": chunk["chunk_id"],
                    "page": chunk["page"],
                }
                for chunk in chunks
            ],
        )


def _load_source_chunks() -> list[dict]:
    chunks = _chunk_faq(FAQ_PATH)

    if chunks:
        return chunks

    return _chunks_from_legacy_chroma()


def _chunks_from_legacy_chroma() -> list[dict]:
    sqlite_path = LEGACY_CHROMA_DIR / "chroma.sqlite3"

    if not sqlite_path.exists():
        return []

    try:
        client = PersistentClient(path=str(LEGACY_CHROMA_DIR))
        collection = client.get_collection(name=LEGACY_COLLECTION)
        raw = collection.get(include=["documents", "metadatas"])
    except Exception:
        return []

    chunks = []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    for index, content in enumerate(documents):
        if not content:
            continue

        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        chunks.append(
            {
                "chunk_id": _as_int(metadata.get("chunk_id")) or index,
                "page": _as_int(metadata.get("page")) or 0,
                "content": str(content).strip(),
            }
        )

    return chunks


def _chunk_faq(pdf_path: Path) -> list[dict]:
    if not pdf_path.is_file():
        return []

    try:
        import pymupdf
    except ImportError:
        return []

    chunks = []
    chunk_id = 0
    document = pymupdf.open(pdf_path)

    try:
        for page_index, page in enumerate(document):
            page_text = (page.get_text() or "").strip()

            if not page_text:
                continue

            for text in _split_text(page_text):
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_index,
                        "content": text,
                    }
                )
                chunk_id += 1
    finally:
        document.close()

    return chunks


def _split_text(text: str) -> list[str]:
    pieces = _recursive_split(text.strip(), SEPARATORS)

    if not pieces:
        return []

    if len(pieces) == 1:
        return pieces

    merged = [pieces[0]]

    for piece in pieces[1:]:
        previous = merged[-1]

        if len(previous) + len(piece) <= CHUNK_SIZE:
            merged[-1] = previous + piece
        else:
            overlap = previous[-CHUNK_OVERLAP:] if len(previous) > CHUNK_OVERLAP else previous
            merged.append((overlap + piece).strip())

    return [item.strip() for item in merged if item.strip()]


def _recursive_split(text: str, separators: list[str]) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []

    if not separators:
        return [
            text[index:index + CHUNK_SIZE]
            for index in range(0, len(text), CHUNK_SIZE)
        ]

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        return [
            text[index:index + CHUNK_SIZE]
            for index in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)
        ]

    parts = text.split(separator)
    chunks: list[str] = []
    current = ""

    for index, part in enumerate(parts):
        fragment = part if index == len(parts) - 1 else part + separator

        if current and len(current) + len(fragment) > CHUNK_SIZE:
            chunks.extend(_recursive_split(current, remaining))
            current = fragment
        else:
            current += fragment

    if current:
        chunks.extend(_recursive_split(current, remaining))

    return [chunk for chunk in chunks if chunk.strip()]


def _parse_query_result(raw: dict) -> list[dict]:
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results = []

    for index, content in enumerate(documents):
        if not content:
            continue

        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}
        distance = distances[index] if index < len(distances) else None

        results.append(
            {
                "chunk_id": _as_int(metadata.get("chunk_id")),
                "content": str(content).strip(),
                "page": _as_int(metadata.get("page")),
                "score": (
                    float(1 - distance)
                    if distance is not None
                    else None
                ),
            }
        )

    return results


def _bm25_rerank(query: str, results: list[dict]) -> list[dict]:
    documents = [item["content"] for item in results]
    lexical_scores = _bm25_scores(query, documents)
    dense_scores = [
        float(item["score"] or 0)
        for item in results
    ]
    lexical_norm = _minmax(lexical_scores)
    dense_norm = _minmax(dense_scores)

    for item, dense, lexical in zip(results, dense_norm, lexical_norm):
        item["score"] = (0.6 * dense) + (0.4 * lexical)

    results.sort(key=lambda item: item["score"] or 0, reverse=True)
    return results


def _bm25_scores(
    query: str,
    documents: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    tokenized_docs = [_tokenize(doc) for doc in documents]
    query_tokens = _tokenize(query)
    n = len(tokenized_docs)

    if n == 0 or not query_tokens:
        return [0.0] * n

    avgdl = sum(len(doc) for doc in tokenized_docs) / n
    document_frequency: Counter[str] = Counter()

    for doc in tokenized_docs:
        document_frequency.update(set(doc))

    scores = []

    for doc in tokenized_docs:
        tf = Counter(doc)
        doc_len = len(doc) or 1
        score = 0.0

        for token in query_tokens:
            if token not in tf:
                continue

            df = document_frequency[token]
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            denom = tf[token] + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (tf[token] * (k1 + 1)) / denom

        scores.append(score)

    return scores


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []

    low = min(values)
    high = max(values)

    if high == low:
        return [1.0 for _ in values]

    return [(value - low) / (high - low) for value in values]


def _tokenize(text: str) -> list[str]:
    return re.findall(
        r"[0-9A-Za-z\u0600-\u06FF]+",
        text.lower(),
    )


def _as_int(value) -> int | None:
    if value is None or value == "":
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


_retriever: KnowledgeRetriever | None = None


def get_knowledge_retriever(
    gemini: GeminiClient,
) -> KnowledgeRetriever:
    global _retriever

    if _retriever is None:
        _retriever = KnowledgeRetriever(gemini)

    return _retriever
