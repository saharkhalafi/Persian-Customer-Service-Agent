from pathlib import Path
import json

import numpy as np

from app.retrieval.knowledge_retriever import _chunk_faq
from app.core.gemini import GeminiClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAQ_PATH = PROJECT_ROOT / "data" / "FAQ.pdf"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "colab_artifacts"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_EMBEDDINGS = OUTPUT_DIR / "chunk_embeddings.npy"
OUTPUT_CHUNKS = OUTPUT_DIR / "chunks.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FAQ and creating chunks...")

    chunks = _chunk_faq(FAQ_PATH)

    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        raise RuntimeError("No chunks were created.")

    # IMPORTANT:
    # _chunk_faq() returns:
    # {
    #     "chunk_id": ...,
    #     "page": ...,
    #     "content": ...
    # }
    texts = [chunk["content"] for chunk in chunks]

    print("Creating Gemini document embeddings...")

    gemini = GeminiClient()

    embeddings = gemini.embed(
        texts,
        task_type="RETRIEVAL_DOCUMENT",
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"Embedding count mismatch: "
            f"{len(embeddings)} embeddings for "
            f"{len(chunks)} chunks."
        )

    embeddings_array = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print(f"Embedding shape: {embeddings_array.shape}")

    expected_shape = (len(chunks), 768)

    if embeddings_array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{embeddings_array.shape}, "
            f"expected {expected_shape}"
        )

    # Save embeddings
    np.save(OUTPUT_EMBEDDINGS, embeddings_array)
    np.save(RESULTS_DIR / "chunk_embeddings.npy", embeddings_array)

    # Save chunks
    output_chunks = []

    for chunk in chunks:
        output_chunks.append(
            {
                "chunk_id": int(chunk["chunk_id"]),
                "page": int(chunk["page"]),
                "content": chunk["content"],
            }
        )

    with open(
        OUTPUT_CHUNKS,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output_chunks,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Embeddings : {OUTPUT_EMBEDDINGS}")
    print(f"Chunks     : {OUTPUT_CHUNKS}")
    print(f"Shape      : {embeddings_array.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()