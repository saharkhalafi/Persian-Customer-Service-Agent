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

6. **Production API, observability, and latency**  
   - Versioned FastAPI: `POST /api/v1/chat` runs the real Agent pipeline (not a demo stub)  
   - Local testing: `http://localhost:8000/docs` (Swagger) and `/redoc`  
   - Auth/context: `X-Customer-ID`; optional `X-Request-ID` / `X-Trace-ID` propagated through Agent, LLM, tools, Product Search, and DB logs  
   - Structured JSON logs at layer boundaries (`request_completed`, `agent_tool_selection`, `tool_execution`, `llm_call`, `product_search_completed`)  
   - Safe error envelope (no stack traces / SQL / keys / prompts); ranking and tool-selection behavior unchanged  
   - Liveness `GET /health` (process only) and readiness `GET /ready` (PostgreSQL ping; LLM is not required because Product Search already has a deterministic fallback)  
   - In-process chat rate limit (default 20 req / 60s per `X-Customer-ID`). Multi-instance production should replace this with a shared backend; no Redis is required for local use  
   - Explicit timeouts: Gemini HTTP (`GEMINI_TIMEOUT_MS`, default 45s, max 2 attempts), Product Search LLM (`LLM_TIMEOUT_SECONDS` → deterministic fallback), PostgreSQL `connect_timeout` + `statement_timeout`, pooled connections with `pool_pre_ping`  
   - Conversation context: `/api/v1/chat` loads recent stored turns for the authenticated customer so follow-ups like «قیمتش چنده؟» stay in the same thread  
   - Grounding: answers may only use product / order / FAQ evidence returned by tools; missing price, stock, URL, status, or dates are not invented  

   Healthy-path latency (Gemini reachable, `gemini-2.5-flash`, thinking disabled):

| Path | Before | After | Change |
|------|--------|-------|--------|
| Greeting (no tool) | 39.4s | **1.6s** | ~25× |
| Product search | 22.3s | **4.1s** | ~5.4× |
| Order summary | 2.9s | **2.1s** | ~26% |

   - Main win: disable Gemini thinking (`AGENT_THINKING_BUDGET=0`); also shorter system prompt, knowledge/metadata caches, parallel independent tools, and `idx_products_brand_lower` after `EXPLAIN ANALYZE`  
   - A 30s+ `/chat` response with a valid `trace_id` is usually **Gemini TLS retries** (`SSL: UNEXPECTED_EOF_WHILE_READING` on Windows), not Product Search. Example: trace `cacdc502…` spent **53s** on the first LLM call (3 SSL retries) then **1.5s** search + **2.1s** final answer  

7. **Retrieval Performance Comparison**  

| Configuration                              | Embedding Model                          | Re-ranker / Fusion                     | Hit Rate @3 | MRR @3 | Precision @3 | Recall @3 | Key Observations / Notes                                      |
|--------------------------------------------|------------------------------------------|----------------------------------------|-------------|--------|--------------|-----------|----------------------------------------------------------------|
| Baseline                                   | paraphrase-multilingual-MiniLM-L12-v2   | —                                      | 0.667       | 0.578  | 0.222        | 0.667     | Basic multilingual model – moderate performance                |
| Improved Dense Retrieval                   | multilingual-e5-large-instruct          | —                                      | 0.867       | 0.722  | 0.289        | 0.867     | Significant gains in recall and ranking quality                |
| Dense + Re-ranking (English-oriented)      | multilingual-e5-large-instruct          | ms-marco-MiniLM-L-12-v2                | 0.533       | 0.344  | 0.178        | 0.533     | Performance degradation due to poor multilingual support       |
| Dense + Re-ranking                         | multilingual-e5-large-instruct          | mxbai-rerank-large-v1                  | 0.933       | 0.811  | 0.311        | 0.933     | Strong earlier configuration; not the current best             |
| **Best Final Configuration**               | **gemini-embedding-001**                | **BM25 hybrid (0.6 dense + 0.4 lexical)** | **1.000 (100%)** | **0.975 (97.5%)** | **0.383 (38.3%)** | **0.975 (97.5%)** | Colab Gemini + BM25; highest Hit@3, MRR, and Recall@3 |


8. **Generation Evaluation**  
   - Semantic metric (most important):  
     - **BERTScore F1** → **71.8%**  
       (very strong for Persian RAG – shows excellent meaning preservation)

9. **Later evaluation: Gemini embedding + BM25 (Colab) — current best retrieval**  
   - Retrieval setup: `gemini-embedding-001` dense retrieval + BM25 hybrid (0.6 dense + 0.4 lexical on top-10, final k=3)  
   - Retrieval metrics:

| Metric | Score |
|--------|-------|
| Hit@3 | **1.000 → 100%** |
| MRR | **0.975 → 97.5%** |
| Precision@3 | **0.383 → 38.3%** |
| Recall@3 | **0.975 → 97.5%** |

   - Generation quality on the FAQ eval set (20 questions), scored with Persian BERTScore (`bert-base-multilingual-cased`):

| Metric | Score |
|--------|-------|
| Precision | 0.7229 |
| Recall | 0.8240 |
| F1 | 0.7692 |
| F1 (%) | **76.92%** |

   - These Colab numbers are **in addition to** the earlier 71.8% BERTScore result above (e5 + mxbai re-ranker). They do not replace it.

10. **Tool-calling evaluation**  
   - Dataset: `Evaluate_data/tool_eval_data.json`  
   - Latest run saved to `evaluation/results/tool_eval_results.json`  
   - **Tool-selection tests used 50 samples**, not 10.  
   - Full suite: **150** cases across categories (tool selection 50, tool argument 25, multi-tool 20, no-tool 15, guardrail 20, security 10, conversation memory 10)

| Metric | Score |
|--------|-------|
| Tool selection accuracy | **83.33%** |
| Tool count accuracy | **86.00%** |
| Overall passed | **125 / 150** |

| Category | Passed | Total | Accuracy |
|----------|--------|-------|----------|
| tool_selection | 46 | 50 | 92.00% |
| tool_argument | 21 | 25 | 84.00% |
| multi_tool | 7 | 20 | 35.00% |
| no_tool | 13 | 15 | 86.67% |
| guardrail | 19 | 20 | 95.00% |
| security | 10 | 10 | 100.00% |
| conversation_memory | 9 | 10 | 90.00% |

Per-tool accuracy (a case counts for a tool when that tool is in `expected_tools`; the **tool-selection category itself is 50 samples**, not 10):

| Tool | Passed | Total | Accuracy |
|------|--------|-------|----------|
| get_conversation_history | 8 | 8 | 100.00% |
| get_customer_profile | 8 | 8 | 100.00% |
| get_purchased_products | 8 | 8 | 100.00% |
| get_order_status | 10 | 11 | 90.91% |
| search_knowledge_base | 9 | 10 | 90.00% |
| search_old_conversations | 8 | 9 | 88.89% |
| get_customer_order_summary | 6 | 7 | 85.71% |
| search_customer_orders | 5 | 6 | 83.33% |
| get_latest_order | 8 | 10 | 80.00% |
| get_order_details | 8 | 10 | 80.00% |
| get_order_history | 9 | 12 | 75.00% |
| get_product_purchase_history | 7 | 10 | 70.00% |
| search_products | 8 | 13 | 61.54% |

11. **Isolated Agent tool-selection + multi-tool evaluation**  
    - Focused 56-case benchmark (does **not** replace the 150-case suite above)  
    - Dataset: `evaluation/datasets/agent_tool_selection_eval.json`  
    - Runner: `python -m evaluation.agent_tool_selection_eval`  
    - Latest results: `evaluation/results/agent_tool_selection_eval.json`  
    - Report: `evaluation/results/agent_tool_selection_report.md`  
    - Covers product intent, order, knowledge/FAQ, no-tool, multi-tool, clarification, and security  
    - Verdict: **READY**

| Metric | Score |
|--------|-------|
| Tool selection accuracy | **89.29%** |
| No-tool accuracy | **100%** |
| Multi-tool exact-match accuracy | **87.5%** |
| `search_products` argument correctness | **100%** |
| Unnecessary tool-call rate | **0%** |
| Missing tool-call rate | **10.71%** |
| Clarification accuracy | **100%** |
| Security accuracy | **100%** |
| RequestContext accuracy | **100%** |
| Overall passed | **50 / 56** |

Per-tool precision / recall / F1:

| Tool | P | R | F1 |
|------|---|---|-----|
| search_products | 1.00 | 0.95 | **0.98** |
| search_knowledge_base | 1.00 | 0.91 | **0.95** |
| get_order_status | 1.00 | 0.67 | 0.80 |
| get_latest_order | 1.00 | 0.50 | 0.67 |
| get_order_history | 1.00 | 1.00 | 1.00 |
| get_order_details | 1.00 | 1.00 | 1.00 |
| get_purchased_products | 1.00 | 1.00 | 1.00 |
| search_customer_orders | 1.00 | 1.00 | 1.00 |

Multi-tool: **7 / 8** set-exact match (0 extra tools, 1 missing `get_order_status`). Clarification: **8 / 8**. Security: no `customer_id` / SQL / filters / Product Search config leaked; scoped `RequestContext` held.

Failed cases (misses only): P-04 price-constrained perfume query skipped `search_products`; O-02 / O-03 / O-06 answered without `get_latest_order`; K-01 skipped FAQ retrieval for returns; M-06 called `get_latest_order` but not `get_order_status`.

Comparison with the previous 150-case `tool_eval` (scores are not interchangeable; this set is smaller and more focused):

| | Previous (`tool_eval`) | This suite |
|--|------------------------|------------|
| Samples | 150 | 56 |
| Selection | 83.33% | **89.29%** |
| Multi-tool | 35.0% | **87.5%** |
| No-tool | 86.67% | **100%** |
| Security | 100% | **100%** |
| `search_products` | 61.54% hit-when-expected | F1 **0.98** |

## Key Achievements

- **Best retrieval: Gemini + BM25**: Hit@3 **100%**, MRR **97.5%**, Precision@3 **38.3%**, Recall@3 **97.5%**  
- **Earlier re-ranker (kept for comparison)**: `mixedbread-ai/mxbai-rerank-large-v1` with e5 embeddings reached Hit@3 0.933 / MRR 0.811 — no longer the best  
- **Solid semantic quality**: BERTScore F1 = 71.8% (strong meaning preservation even with different wording)  
- **Colab Gemini + BM25 generation**: BERTScore F1 = 76.92% (Precision 0.7229, Recall 0.8240) on 20 FAQ questions  
- **Tool calling**: 125/150 cases passed; tool-selection category = **50 samples** (not 10); selection accuracy 83.33%, count accuracy 86.00%  
- **Focused Agent tool-selection eval**: 50/56 passed; selection **89.29%**, multi-tool **87.5%**, no-tool / clarification / security **100%**; `search_products` F1 **0.98**; verdict **READY**  
- **Very low hallucination** thanks to strong retriever + re-ranking + strict prompt  
- **Fully reproducible** local pipeline with CLI interface  
- **Manual + automated evaluation** (human-labeled chunks + JSON export)  
- **Real-world focus**: built for actual Modiseh FAQ content
- **UI**: Streamlit 

## Tech Stack

- **Framework**: LangChain  
- **Vector DB**: Chroma (persistent, local)  
- **Embeddings**: gemini-embedding-001 (best); intfloat/multilingual-e5-large-instruct (earlier pipeline)  
- **Retrieval fusion / re-ranker**: BM25 hybrid 0.6 dense + 0.4 lexical (best); mixedbread-ai/mxbai-rerank-large-v1 (earlier)  
- **LLM**: Google Gemini (gemini-2.0-flash)  
- **Evaluation**: BERTScore, ROUGE, BLEU, custom retrieval metrics, tool-calling accuracy (`evaluation/tool_eval.py`), isolated Agent tool-selection + multi-tool eval (`evaluation/agent_tool_selection_eval.py`)  
- **Tools**: pandas, sentence-transformers, openpyxl, rouge-score, nltk

### Production API notes

Secrets (`DATABASE_URL`, `GEMINI_API_KEY`) come from the environment only. Optional knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_TIMEOUT_MS` | `45000` | Gemini HTTP timeout |
| `GEMINI_RETRY_ATTEMPTS` | `2` | Transient LLM/network retries only (max 3) |
| `LLM_TIMEOUT_SECONDS` | `20` | Product Search LLM; falls back to deterministic ranking |
| `DB_CONNECT_TIMEOUT_SECONDS` | `5` | PostgreSQL connect timeout |
| `DB_STATEMENT_TIMEOUT_MS` | `30000` | PostgreSQL statement timeout |
| `CHAT_RATE_LIMIT_REQUESTS` | `20` | In-process `/api/v1/chat` limit per customer |
| `CHAT_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |
| `CORS_ALLOWED_ORIGINS` | localhost:8000 / 8501 | Allowed browser origins |
| `LLM_METADATA_FALLBACK_ENABLED` | `true` | Product Search LLM metadata fallback only |
| `LLM_CONFIDENCE_THRESHOLD` | `0.7` | Merge threshold for Product Search LLM metadata |
| `LLM_RERANK_ENABLED` | `false` | Keep off; do not enable rerank |
| `PRODUCT_NAME_MATCHING_ENABLED` | `false` | Keep off; do not enable name matching |

Chat errors always look like `{ "error": { "code", "message", "trace_id" } }` with `400/401/403/404/429/500/503`. Use `trace_id` to find server logs. The Agent never accepts `customer_id` from the model; identity is only `X-Customer-ID` → `RequestContext`.

`GET /metrics` exposes in-process Prometheus text (API, Agent, Product Search, LLM). OpenTelemetry is **not** bundled; span names and `trace_id` / optional W3C `traceparent` are ready for a later exporter.

`POST /api/v1/feedback` stores thumbs-up/down (`rating`, optional `conversation_id` / `message_id`) for the header customer. Chat still does not return server-issued conversation IDs.

### Local developer experience

Copy `.env.example` to `.env`. Do not commit secrets.

1. **Environment variables required:** `DATABASE_URL`, `GEMINI_API_KEY`. Other knobs are optional (see `.env.example`).
2. **Database:** PostgreSQL with the existing `orders` / `products` / customer tables. Schema changes (including `conversation_messages` and `message_feedback`) are applied with Alembic, not on API startup. FAQ retrieval uses the existing local knowledge store, not a new vector database.
3. **Start the API (development only):** `uvicorn app.main:app --reload --port 8000`
4. **Swagger:** http://localhost:8000/docs
5. **Health:** `GET http://localhost:8000/health`
6. **Readiness:** `GET http://localhost:8000/ready` (PostgreSQL ping; Gemini is not required)
7. **Example chat:**

```bash
curl -s http://localhost:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -H "X-Customer-ID: 9206288" ^
  -H "X-Trace-ID: local-manual-1" ^
  -d "{\"message\": \"کرم ضد چروک پرایم دارید؟\"}"
```

8. **Inspect logs:** uvicorn writes structured JSON to stdout. Filter on `trace_id` (same value as the response / `X-Trace-ID`). Events include `request_completed`, `agent_tool_selection`, `tool_execution`, `llm_call`, `product_search_completed`, `feedback_recorded`. Secrets and raw SQL are not logged.
9. **Error cases:** omit `X-Customer-ID` (401), send `{"message":""}` (400), exceed the chat rate limit (429), stop PostgreSQL and call `/ready` (503).

Automated smoke (no live Gemini): `python -m pytest tests/test_api_smoke.py tests/test_api_observability.py`. Optional live API smoke against a running server: `set RUN_LIVE_SMOKE=1` then `python -m evaluation.local_smoke`.

### Database migrations (Alembic)

The API does **not** create or alter tables at startup. Apply schema with Alembic against the same `DATABASE_URL` the app uses.

```bash
pip install -r requirements-dev.txt
alembic upgrade head
alembic current
alembic history
alembic revision -m "describe the change"
alembic downgrade -1
```

`0001_initial` is safe on an existing catalog: it creates `users` / `products` / `orders` only when they are missing, and always ensures `conversation_messages`, `message_feedback`, and `idx_products_brand_lower`. Downgrade of that revision drops only the application-owned tables/indexes; it does not drop catalog data.

### Docker / Compose

Production image: `Dockerfile` (no `--reload`, non-root, env-based config). Compose starts Postgres, runs `alembic upgrade head`, then the API.

```bash
docker compose up --build
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

Set secrets in `.env` or the shell (`GEMINI_API_KEY`, `COMPOSE_POSTGRES_PASSWORD`). Compose Postgres does not reuse host `DATABASE_URL` / `POSTGRES_USER`. After changing Compose DB credentials, recreate the volume: `docker compose down -v`. Do not bake secrets into the image. Production: `ENVIRONMENT=production`.

CI (GitHub Actions) runs pytest, Alembic upgrade/downgrade/upgrade against a service Postgres, and `docker build`. It uses a placeholder `GEMINI_API_KEY` and does not need a real Gemini credential.

**Future OpenTelemetry step:** keep the current `trace_id` ContextVar and `/metrics` names; add an OTLP exporter that maps `http_request` / `agent` / `llm` / `tool` / `product_search` / `database` log events to spans. Do not introduce Kafka, Redis, Celery, or a second API framework for that.

## Demo
![Screenshot](https://github.com/saharkhalafi/persian-rag-faq/blob/main/Evaluate_data/web%20UI.png)
