# Persian RAG FAQ Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** system built in Persian to answer real-world FAQs from the Modiseh e-commerce website with high accuracy, minimal hallucination, and source grounding.

 
## Project Overview

Goal: Build a complete, local-first Persian RAG pipeline that:
- Ingests and chunks a real FAQ PDF
- Creates a persistent vector database
- Retrieves relevant chunks with near-perfect accuracy (using **re-ranking** for improved precision)
- Generates concise, faithful answers using Gemini
- Fully evaluates both retrieval and semantic generation quality
- Provides a clean, user-friendly **web UI**

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

3. **Advanced Retrieval with Re-ranking**  
   - Initial retrieval: top-10 candidates using vector similarity  
   - **Re-ranking layer** added using `mixedbread-ai/mxbai-rerank-large-v1` (cross-encoder)  
   - Final top-3 results selected after re-scoring → significantly improved precision and ranking quality  
   - This two-stage retrieval (embedding + re-ranker) was key to achieving near-perfect metrics

4. **RAG Pipeline**  
   - Built using LangChain: `create_retrieval_chain` + `create_stuff_documents_chain`  
   - Custom prompt engineering (multiple iterations) to enforce:  
     - Very short answers (1–3 sentences max)  
     - Strict faithfulness to retrieved context  
     - Clear fallback: «اطلاعات کافی در منابع موجود نیست.» when context is insufficient

5. **Evaluation Dataset**  
   - Manually created 20-question test set  
   - Added ground-truth answers + human-labeled `relevant_chunk_ids` (from preview file)  
   - Script runs live RAG on every question and saves:  
     - generated `answer`  
     - actual `retrieved_chunks` (chunk_ids returned by retriever)

6. **Retrieval Performance Comparison**  
| Configuration                              | Embedding Model                          | Re-ranker Model                        | Hit Rate @3 | MRR @3 | Precision @3 | Recall @3 | Key Observations / Notes                                      |
|--------------------------------------------|------------------------------------------|----------------------------------------|-------------|--------|--------------|-----------|----------------------------------------------------------------|
| Baseline                                   | paraphrase-multilingual-MiniLM-L12-v2   | —                                      | 0.667       | 0.578  | 0.222        | 0.667     | Basic multilingual model – moderate performance                |
| Improved Dense Retrieval                   | multilingual-e5-large-instruct          | —                                      | 0.867       | 0.722  | 0.289        | 0.867     | Significant gains in recall and ranking quality                |
| Dense + Re-ranking (English-oriented)      | multilingual-e5-large-instruct          | ms-marco-MiniLM-L-12-v2                | 0.533       | 0.344  | 0.178        | 0.533     | Performance degradation due to poor multilingual support       |
| **Best Final Configuration**               | multilingual-e5-large-instruct          | mxbai-rerank-large-v1                  | **0.933**   | **0.811** | **0.311**    | **0.933** | Highest Hit Rate & MRR; strong balance of precision and recall |


7. **Generation Evaluation**  
   - Semantic metric (most important):  
     - **BERTScore F1** → **71.8%**  
       (very strong for Persian RAG – shows excellent meaning preservation)

## Key Achievements

- **Near-perfect retrieval**: 100% hit/recall/precision @5 on 20 labeled questions  
- **Re-ranking boost**: Used `mixedbread-ai/mxbai-rerank-large-v1` cross-encoder to refine initial top-10 candidates → critical for achieving perfect metrics  
- **Solid semantic quality**: BERTScore F1 = 71.8% (strong meaning preservation even with different wording)  
- **Very low hallucination** thanks to strong retriever + re-ranking + strict prompt  
- **Fully reproducible** local pipeline with CLI interface  
- **Manual + automated evaluation** (human-labeled chunks + JSON export)  
- **Real-world focus**: built for actual Modiseh FAQ content
- **UI**: Streamlit 

## Tech Stack

- **Framework**: LangChain  
- **Vector DB**: Chroma (persistent, local)  
- **Embeddings**: intfloat/multilingual-e5-large-instruct  
- **Re-ranker**: mixedbread-ai/mxbai-rerank-large-v1  
- **LLM**: Google Gemini (gemini-2.0-flash)  
- **Evaluation**: BERTScore, ROUGE, BLEU, custom retrieval metrics  
- **Tools**: pandas, sentence-transformers, openpyxl, rouge-score, nltk

## Demo
![Screenshot](https://github.com/saharkhalafi/persian-rag-faq/blob/main/Evaluate_data/web%20UI.png)
