
import os
import json
import logging
import pandas as pd
from typing import List, Dict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import torch
from sentence_transformers import CrossEncoder

# ──── Cell 4: Configuration ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RAG-EVAL-FLOW")

# Paths – change only if your folder structure is different
BASE_FOLDER    = "/RAG_Project"
PDF_PATH       = "/FAQ.pdf"
EXCEL_PATH     = "/rag_eval_dataset_modiseh.xlsx"
CHROMA_PERSIST = "/RAG_Project/chroma_db_eval/"

# Use local folder to avoid Drive write-permission issues
LOCAL_CHROMA   = "/RAG_Project/local_chroma_db2"
CHUNK_SIZE       = 600
CHUNK_OVERLAP    = 100
INITIAL_K        = 10
FINAL_K          = 3

EMBEDDING_MODEL  = "intfloat/multilingual-e5-large-instruct"
RERANKER_MODEL   = "mixedbread-ai/mxbai-rerank-large-v1"

os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(LOCAL_CHROMA, exist_ok=True)

# ──── chuncking function ────────────────────────────────────────────────────────
def chunk_document(pdf_path: str) -> List[Document]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF پیدا نشد: {pdf_path}")

    logger.info(f"بارگذاری PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", ". ", "!", "؟", " ", ""],
        keep_separator=True,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    logger.info(f"→ تعداد چانک‌ها: {len(chunks)}")

    # important to assign unique chunk_id to each chunk for later reference in evaluation
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks

# ──── preview of chunks ──────────────────────────────────────
def print_chunks_preview(chunks: List[Document], preview_len=220):
    print("\n" + "═" * 80)
    print("CHUNK PREVIEW – از این idها برای برچسب‌گذاری relevant_chunk_ids استفاده کن")
    print("═" * 80 + "\n")

    for chunk in chunks:
        cid = chunk.metadata["chunk_id"]
        text = chunk.page_content.strip().replace("\n", " ")
        preview = (text[:preview_len] + "...") if len(text) > preview_len else text
        print(f"[{cid:3d}]  {preview}")

    # save preview to txt file for easier reference
    out_file = os.path.join(BASE_FOLDER, "chunks_preview.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(f"[{chunk.metadata['chunk_id']:3d}] {chunk.page_content}\n\n")
    print(f"\nچانک‌ها در فایل ذخیره شدند → {out_file}\n")

# ──── create or load DB──────────────────────────────────────────────────
def get_vector_db(chunks: List[Document] = None, force_recreate: bool = False) -> Chroma:
    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},  # یا "cuda" اگر GPU داری
        encode_kwargs={"normalize_embeddings": True}
    )

    if force_recreate and os.path.exists(LOCAL_CHROMA):
        import shutil
        shutil.rmtree(LOCAL_CHROMA)
        print("→ فولدر Chroma قبلی حذف شد (force recreate)")

    if os.path.exists(LOCAL_CHROMA) and os.listdir(LOCAL_CHROMA):
        logger.info("لود دیتابیس موجود")
        return Chroma(persist_directory=LOCAL_CHROMA, embedding_function=embedder)

    if not chunks:
        raise ValueError("چانک‌ها موجود نیستند و دیتابیس هم پیدا نشد.")

    logger.info("ساخت دیتابیس جدید...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=LOCAL_CHROMA,
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info("دیتابیس ساخته شد")
    return db

# ──── Load eval dataset ───────────────────────────────────────────────
def load_eval_data(excel_path: str) -> List[Dict]:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"فایل اکسل پیدا نشد: {excel_path}")

    print(f"بارگذاری دیتاست ارزیابی: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0, engine="openpyxl", dtype=str)

    df.columns = df.columns.str.strip().str.lower()

    data = []
    for _, row in df.iterrows():
        item = {
            "question": str(row.get("question", "")).strip(),
            "reference_answer": str(row.get("reference_answer", "")).strip(),
            "relevant_keywords": [k.strip() for k in str(row.get("relevant_keywords", "")).split(",") if k.strip()],
            "relevant_section": str(row.get("relevant_section", "")).strip(),
            "difficulty": str(row.get("difficulty", "medium")).strip(),
            "relevant_chunks": []
        }

        ids_str = str(row.get("relevant_chunk_ids", "")).strip()
        if ids_str:
            try:
                item["relevant_chunks"] = [int(x.strip()) for x in ids_str.split(",") if x.strip().isdigit()]
            except:
                print(f"هشدار: نمی‌توان chunk ids را پارس کرد → {ids_str}")

        data.append(item)

    print(f"→ تعداد سوالات: {len(data)}")
    return data

# ──── evaluation function ──────────────────────────────────────────────────────────
def evaluate_retrieval(db: Chroma, eval_data: List[Dict], k: int = 3, use_reranking: bool = False) -> Dict[str, float]:
    initial_k = INITIAL_K if use_reranking else k
    retriever = db.as_retriever(search_kwargs={"k": initial_k})

    reranker = None
    if use_reranking:
        reranker = CrossEncoder(RERANKER_MODEL)
        print(f"استفاده از re-ranker: {RERANKER_MODEL}")

    hit, mrr, prec, rec = [], [], [], []

    for item in eval_data:
        q = item["question"]
        relevant = set(item.get("relevant_chunks", []))

        if not relevant:
            continue

        docs = retriever.invoke(q)

        if use_reranking and reranker:
            pairs = [[q, d.page_content] for d in docs]
            scores = reranker.predict(pairs)
            ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            final_docs = [docs[i] for i in ranked_idx[:k]]
        else:
            final_docs = docs[:k]

        retrieved_ids = [d.metadata.get("chunk_id", -1) for d in final_docs]

        hits = len(set(retrieved_ids) & relevant)

        hit.append(1 if hits > 0 else 0)

        rank = 0
        for r, rid in enumerate(retrieved_ids, 1):
            if rid in relevant:
                rank = 1 / r
                break
        mrr.append(rank)

        p = hits / len(retrieved_ids) if retrieved_ids else 0
        r = hits / len(relevant) if relevant else 0
        prec.append(p)
        rec.append(r)

    if not hit:
        return {"error": "هیچ آیتم معتبری با relevant chunks پیدا نشد"}

    return {
        f"HitRate@{k}": round(sum(hit)/len(hit), 3),
        f"MRR@{k}": round(sum(mrr)/len(mrr), 3),
        f"Precision@{k}": round(sum(prec)/len(prec), 3),
        f"Recall@{k}": round(sum(rec)/len(rec), 3),
    }

# ──── run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # PDF chunking + preview
    chunks = chunk_document(PDF_PATH)
    print_chunks_preview(chunks)

    #  force_recreate=True for delete pervious DB and create new one
    db = get_vector_db(chunks, force_recreate=False)  
    eval_dataset = load_eval_data(EXCEL_PATH)

    # Without reranking evaluation
    print("\n" + "═"*75)
    print("BASIC RETRIEVAL")
    print("═"*75)
    basic = evaluate_retrieval(db, eval_dataset, k=FINAL_K, use_reranking=False)
    for k,v in basic.items():
        print(f"{k:16} : {v:.3f}")

    # reranking evaluation
    print("\n" + "═"*75)
    print("RE-RANKED RETRIEVAL")
    print("═"*75)
    reranked = evaluate_retrieval(db, eval_dataset, k=FINAL_K, use_reranking=True)
    for k,v in reranked.items():
        print(f"{k:16} : {v:.3f}")


    results = {"basic": basic, "reranked": reranked}
    save_path = os.path.join(BASE_FOLDER, "retrieval_metrics_2026.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nنتایج ذخیره شد → {save_path}")