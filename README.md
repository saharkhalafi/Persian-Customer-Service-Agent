# Persian RAG FAQ Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** system built in Persian to answer real-world FAQs from the Modiseh e-commerce website with high accuracy, minimal hallucination, and source grounding.

## Project Overview

Goal: Build a complete, local-first Persian RAG pipeline that:
- Ingests and chunks a real FAQ PDF
- Creates a persistent vector database
- Retrieves relevant chunks with near-perfect accuracy
- Generates concise, faithful answers using Gemini
- Fully evaluates both retrieval and semantic generation quality

## What I Built – Step by Step

1. **Data Ingestion & Chunking**  
   - Loaded FAQ PDF with `PyPDFLoader`  
   - Split into meaningful chunks (size 600, overlap 100) using `RecursiveCharacterTextSplitter`  
   - Assigned unique `chunk_id` to every chunk's metadata  
   - Generated `chunks_preview.txt` for easy manual relevance labeling

2. **Local Vector Database**  
   - Persistent **Chroma** vector store with cosine HNSW index  
   - Embeddings: `intfloat/multilingual-e5-large-instruct` (strong multilingual & Persian support)  
   - Fully local – no cloud vector DB needed (only Gemini API for generation)

3. **RAG Pipeline**  
   - Built using LangChain: `create_retrieval_chain` + `create_stuff_documents_chain`  
   - Custom prompt engineering (multiple iterations) to enforce:  
     - Very short answers (1–3 sentences max)  
     - Strict faithfulness to retrieved context  
     - Clear fallback message when context is insufficient

4. **Evaluation Dataset**  
   - Manually created 20-question test set  
   - Added ground-truth answers + human-labeled `relevant_chunk_ids` (from preview file)  
   - Wrote script that runs live RAG on every question and saves:  
     - generated `answer`  
     - actual `retrieved_chunks` (chunk_ids returned by retriever)

5. **Retrieval Evaluation**  
   - Compared retrieved chunk IDs against human-labeled relevant IDs  
   - Results (k=5):  
     - **Hit Rate @5** → **100.0%**  
     - **Recall @5** → **100.0%**  
     - **Precision @5** → **100.0%**

6. **Generation Evaluation**  
   - Semantic metric (most important):  
     - **BERTScore F1** → **71.8%**  
       (very strong for Persian RAG – shows excellent meaning preservation)

## Key Achievements

- **Near-perfect retrieval**: 100% hit/recall/precision @5 on 20 labeled questions  
- **Solid semantic quality**: BERTScore F1 = 71.8% (meaning is faithfully transferred)  
- **Very low hallucination** thanks to strong retriever + strict prompt  
- **Fully reproducible** local pipeline with CLI interface  
- **Manual + automated evaluation** (human-labeled chunks + JSON export)  
- **Real-world focus**: built for actual Modiseh FAQ content

## Tech Stack

- **Framework**: LangChain  
- **Vector DB**: Chroma (persistent, local)  
- **Embeddings**: intfloat/multilingual-e5-large-instruct  
- **LLM**: Google Gemini (gemini-2.0-flash)  
- **Evaluation**: BERTScore, ROUGE, BLEU, custom retrieval metrics  
- **Tools**: pandas, sentence-transformers, openpyxl, rouge-score, nltk
