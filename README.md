# 🤖 Persian AI Customer Service Agent

### Production-Oriented RAG & Tool-Calling System for Persian E-Commerce

An end-to-end **Persian AI Customer Service Agent** designed for real-world e-commerce support.

The system combines **Retrieval-Augmented Generation (RAG), LLM tool calling, deterministic product search, customer/order services, conversation memory, security boundaries, observability, automated evaluation, PostgreSQL, Docker, Alembic, and CI/CD** into a production-oriented architecture.

Originally developed as a Persian FAQ RAG chatbot for the Modiseh e-commerce website, the project evolved into a broader **AI customer-service platform** capable of answering knowledge-based questions, retrieving customer and order information, searching products, handling multi-step requests, maintaining conversation context, and generating grounded responses.

> **Core principle:** Use the LLM for reasoning and orchestration, while keeping business-critical operations deterministic, secure, observable, and testable.

---

# ✨ Key Highlights

* 🇮🇷 Persian-first AI customer-service system
* 🧠 Gemini-powered Agent with tool calling
* 🔎 Retrieval-Augmented Generation for FAQ / website knowledge
* 🛍️ Dedicated deterministic product-search engine
* 📦 Customer and order management tools
* 💬 Persistent conversation context and follow-up handling
* 🛡️ Customer-scoped authorization boundaries
* 🚫 No LLM-controlled `customer_id`, SQL, or database filters
* 📊 Retrieval, generation, and agent evaluation pipelines
* 🔬 250-case tool-calling benchmark
* 🎯 100-case focused Agent orchestration benchmark
* ⚡ Significant latency optimization
* 💰 Cost-aware LLM architecture
* 📝 Structured JSON observability
* 🔭 Request tracing and metrics
* 🗄️ PostgreSQL + repository/service architecture
* 🔄 Alembic database migrations
* 🐳 Production Docker image + Docker Compose
* 🔁 GitHub Actions CI
* 🧪 169+ automated tests
* 📚 Fully reproducible local development workflow

---

# 🏗️ System Architecture

![Screenshot](https://github.com/saharkhalafi/Persian-Customer-Service-Agent/blob/main/evaluation/results/architecture%20(2).png) 

---
# 🧠 AI Agent

The Agent uses **Google Gemini** for reasoning and controlled tool orchestration.

The LLM is responsible for:

* Understanding user intent
* Selecting the appropriate tool
* Generating tool arguments
* Coordinating multiple tools
* Handling conversational context
* Generating the final natural-language response

The LLM is **not** responsible for:

* Authorization
* Customer identity
* SQL generation
* Database filtering
* Business-critical calculations
* Product ranking

This separation provides a strong boundary between probabilistic AI behavior and deterministic backend logic.

---

# 🧰 Tool-Based Architecture

The Agent exposes specialized backend capabilities such as:

```text
Customer
├── get_customer_profile
├── get_purchased_products
├── get_product_purchase_history
└── get_customer_order_summary

Orders
├── get_latest_order
├── get_order_status
├── get_order_details
├── get_order_history
└── search_customer_orders

Knowledge
├── search_knowledge_base
└── search_old_conversations

Products
└── search_products

Conversation
└── get_conversation_history
```

Tools follow the architecture:

```text
Agent
  ↓
Tool
  ↓
Service
  ↓
Repository
  ↓
Database
```

This keeps LLM orchestration separate from application logic.

---

# 🔐 Security Architecture

Customer identity is controlled by the backend rather than the LLM.

```text
HTTP Request
     │
     │ X-Customer-ID
     ▼
RequestContext
     │
     ▼
Backend Tools
     │
     ▼
Customer-scoped Repository Queries
```

The model cannot provide or override:

```text
customer_id
SQL
database filters
authorization context
```

The Agent never accepts `customer_id` from model-generated tool arguments.

Customer/order repositories enforce customer scoping at the backend level.

This prevents model-generated arguments from becoming an authorization mechanism.

> For local/trusted environments, `X-Customer-ID` is used as the trusted identity boundary. A real public deployment should replace this with JWT/session authentication or a trusted BFF/API gateway.

---

# 📚 Retrieval-Augmented Generation

The original project started as a Persian FAQ RAG system built around real Modiseh e-commerce FAQ content.

## RAG Pipeline

```text
FAQ PDF
   │
   ▼
PyPDFLoader
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ├── chunk_size = 600
   └── chunk_overlap = 100
   │
   ▼
Unique chunk_id
   │
   ▼
Embeddings
   │
   ▼
Persistent Chroma
   │
   ▼
Dense Retrieval
   │
   ▼
Candidate Ranking / Fusion
   │
   ▼
Relevant Context
   │
   ▼
Gemini
   │
   ▼
Grounded Persian Answer
```

The system was designed to be **local-first**:

* Documents are processed locally
* Vector storage is persistent
* Retrieval does not require a cloud vector database
* Gemini is used for controlled language generation

---

# 🎯 Grounding & Hallucination Control

The generation layer uses strict grounding rules.

The Agent should only answer using evidence returned by:

* Knowledge retrieval
* Product Search
* Customer tools
* Order tools
* Conversation context

If required information is unavailable, the system should not invent it.

For example, the Agent must not hallucinate:

```text
Price
Stock
Product URL
Order status
Delivery date
Customer information
```

For insufficient FAQ context, the RAG pipeline uses a safe fallback such as:

> «اطلاعات کافی در منابع موجود نیست.»

This makes the system suitable for customer-facing scenarios where unsupported answers are more harmful than saying "I don't know."

---

# 🛍️ Product Search Engine

Product Search is intentionally implemented as an **independent retrieval subsystem** rather than being mixed with conversational reasoning.

The Agent only invokes:

```text
search_products(query)
```

The search engine itself owns:

* Query normalization
* Persian metadata extraction
* Attribute canonicalization
* Confidence estimation
* Hard filtering
* Classical relevance ranking
* Top-K retrieval

## Production Search Pipeline

```text
Natural Language Query
        │
        ▼
Query Normalization
        │
        ▼
Deterministic Metadata Extraction
        │
        ▼
Confidence / Ambiguity Gate
        │
        ├── High Confidence
        │        ↓
        │   Canonicalization
        │
        └── Low Confidence
                 ↓
        Optional LLM Metadata Fallback
                 │
                 ▼
        Canonicalization / Validation
                 │
                 ▼
        PostgreSQL Hard Filters
                 │
                 ▼
        Classical Ranking
                 │
                 ▼
               Top-K
```

Current production configuration:

```env
LLM_METADATA_FALLBACK_ENABLED=true
LLM_CONFIDENCE_THRESHOLD=0.7
LLM_RERANK_ENABLED=false
PRODUCT_NAME_MATCHING_ENABLED=false
```

The LLM is used only as a targeted fallback for ambiguous metadata extraction.

It is **not used for SQL generation, authorization, or final product ranking**.

---

# 🔬 Retrieval Experiments

Several retrieval configurations were evaluated before selecting the current architecture.

| Configuration           | Embedding                      | Ranking / Fusion      |    Hit@3 |     MRR@3 |
| ----------------------- | ------------------------------ | --------------------- | -------: | --------: |
| Baseline                | MiniLM multilingual            | —                     |    66.7% |     57.8% |
| Improved Dense          | multilingual-e5-large-instruct | —                     |    86.7% |     72.2% |
| Dense + MS MARCO        | multilingual-e5-large-instruct | ms-marco-MiniLM       |    53.3% |     34.4% |
| Dense + mxbai           | multilingual-e5-large-instruct | mxbai-rerank-large-v1 |    93.3% |     81.1% |
| **Best RAG Experiment** | **gemini-embedding-001**       | **BM25 hybrid**       | **100%** | **97.5%** |

The experiments demonstrated an important engineering principle:

> A more complex model or reranker does not automatically produce a better retrieval system.

The final retrieval configuration was selected based on measured benchmark performance.

---

# 📊 RAG Evaluation

A manually labeled FAQ evaluation dataset was created with:

* User questions
* Reference answers
* Relevant keywords
* Relevant sections
* Ground-truth `chunk_id`s
* Difficulty labels

Retrieval and generation are evaluated independently.

## Best Retrieval Experiment

Using Gemini embeddings + BM25 hybrid retrieval:

| Metric      |      Score |
| ----------- | ---------: |
| Hit@3       | **100.0%** |
| MRR@3       |  **97.5%** |
| Precision@3 |  **38.3%** |
| Recall@3    |  **97.5%** |

## Generation Evaluation

Persian semantic evaluation using multilingual BERTScore:

| Metric    |      Score |
| --------- | ---------: |
| Precision |     72.29% |
| Recall    |     82.40% |
| F1        | **76.92%** |

An earlier e5 + mxbai configuration achieved:

```text
BERTScore F1 = 71.8%
```

These results are preserved as experimental baselines rather than replacing one another.

---

# 🤖 Agent Tool-Calling Evaluation

The project contains a dedicated evaluation framework for testing Agent orchestration rather than only evaluating final text quality.

## 250-Case Evaluation

The full benchmark contains:

```text
Tool Selection       100
Tool Arguments       50
Multi-tool            45
No-tool               15
Guardrails            20
Security              10
Conversation Memory   10
────────────────────────
Total                 250
```

Results:

| Metric                  |         Score |
| ----------------------- | ------------: |
| Tool Selection Accuracy |    **83.33%** |
| Tool Count Accuracy     |    **86.00%** |
| Overall Passed          | **225 / 250** |

Selected category results:

| Category            |    Accuracy |
| ------------------- | ----------: |
| Tool Selection      |  **92.00%** |
| Tool Arguments      |      84.00% |
| Multi-tool          |      35.00% |
| No-tool             |      86.67% |
| Guardrail           |  **95.00%** |
| Security            | **100.00%** |
| Conversation Memory |  **90.00%** |

---

# 🎯 Focused Agent Orchestration Evaluation

A second, more focused benchmark was introduced to evaluate Agent behavior independently from the larger 250-case suite.

Dataset:

```text
evaluation/datasets/agent_tool_selection_eval.json
```

Runner:

```bash
python -m evaluation.agent_tool_selection_eval
```

## Results

100 targeted cases:

| Metric                                 |       Score |
| -------------------------------------- | ----------: |
| Tool Selection Accuracy                |  **89.29%** |
| No-tool Accuracy                       |    **100%** |
| Multi-tool Exact Match                 |   **87.5%** |
| `search_products` Argument Correctness |    **100%** |
| Unnecessary Tool Rate                  |      **0%** |
| Missing Tool Rate                      |      10.71% |
| Clarification Accuracy                 |    **100%** |
| Security Accuracy                      |    **100%** |
| RequestContext Accuracy                |    **100%** |
| Overall Passed                         | **96 / 100** |

Per-tool performance:

| Tool                   | Precision | Recall |       F1 |
| ---------------------- | --------: | -----: | -------: |
| search_products        |      1.00 |   0.95 | **0.98** |
| search_knowledge_base  |      1.00 |   0.91 | **0.95** |
| get_order_status       |      1.00 |   0.67 |     0.80 |
| get_latest_order       |      1.00 |   0.50 |     0.67 |
| get_order_history      |      1.00 |   1.00 | **1.00** |
| get_order_details      |      1.00 |   1.00 | **1.00** |
| get_purchased_products |      1.00 |   1.00 | **1.00** |
| search_customer_orders |      1.00 |   1.00 | **1.00** |

---

# ⚡ Performance & Latency Optimization

The system underwent explicit latency profiling and optimization.

Healthy-path measurements using `gemini-2.5-flash` with thinking disabled:

| Path           | Before |    After | Improvement |
| -------------- | -----: | -------: | ----------: |
| Greeting       |  39.4s | **1.6s** |        ~25× |
| Product Search |  22.3s | **4.1s** |       ~5.4× |
| Order Summary  |   2.9s | **2.1s** |        ~26% |

Major optimizations included:

* Disabled Gemini thinking for the Agent
* Reduced system-prompt overhead
* Knowledge caching
* Metadata caching
* Parallel execution of independent tools
* PostgreSQL indexing
* `idx_products_brand_lower`
* Explicit Gemini timeouts
* Limited transient retries
* Deterministic Product Search fallback

One important diagnostic finding was that occasional 30s+ requests were not caused by Product Search.

For example, a request with trace ID `cacdc502…` spent approximately 53 seconds in the initial Gemini call due to repeated Windows TLS failures:

```text
SSL: UNEXPECTED_EOF_WHILE_READING
```

The actual Product Search and final response stages remained fast.

This trace-based diagnosis prevents optimizing the wrong component.

---

# 💰 Cost-Aware Architecture

The system intentionally avoids unnecessary LLM calls.

LLMs are primarily used for:

```text
Reasoning
Tool Selection
Natural Language Generation
Low-confidence Metadata Fallback
```

Deterministic components handle:

```text
Authorization
Customer Identity
Database Access
Hard Filtering
Product Ranking
Business Rules
```

Estimated Gemini cost based on measured token usage and current list pricing:

| Request Type    | Estimated Cost |
| --------------- | -------------: |
| Simple Greeting |  ~$0.002–0.003 |
| Order Request   |  ~$0.004–0.007 |
| Product Search  |  ~$0.006–0.012 |
| Mixed Traffic   |  ~$0.005–0.008 |

These are planning estimates rather than actual production billing.

Example:

```text
1,000 chats/day  → ~$210/month
5,000 chats/day  → ~$1,050/month
```

The architecture is designed around:

```text
Quality × Latency × Cost
```

rather than maximizing LLM usage.

---

# 💬 Conversation Memory

The Agent supports persistent conversation context.

The chat service loads recent stored turns for the authenticated customer, allowing follow-up queries such as:

```text
User: «کفش نایک مشکی دارید؟»

Agent: ...

User: «قیمتش چنده؟»
```

to remain grounded in the existing conversation.

Conversation persistence is handled through:

```text
Conversation Service
        ↓
Conversation Repository
        ↓
PostgreSQL
```

---

# 📊 Observability

Every important execution boundary can be traced using a shared `trace_id`.

```text
HTTP Request
      ↓
    Agent
      ↓
     LLM
      ↓
    Tool
      ↓
Product Search / RAG
      ↓
   Service
      ↓
  Database
```

Structured JSON events include:

```text
request_completed
request_error
agent_tool_selection
agent_tool_selection_error
tool_execution
agent_completed
llm_call
product_search_completed
feedback_recorded
db_operation
service_operation
```

Tracked metrics include:

* API request count
* API error rate
* API latency
* Agent latency
* Tool execution latency
* LLM latency
* LLM token usage
* Estimated LLM cost
* LLM timeout/failure rate
* Product Search latency
* Product Search empty-result rate
* LLM metadata fallback frequency
* Feedback statistics

Sensitive information is intentionally excluded from logs.

The system does not log:

```text
API keys
Passwords
Database credentials
Raw SQL
Prompts
Stack traces in HTTP responses
Customer-sensitive data
```

OpenTelemetry is not bundled yet, but the current trace/span boundaries are designed to support future integration.

---

# 🌐 Production API

The application exposes a versioned FastAPI interface.

```text
POST /api/v1/chat
POST /api/v1/feedback

GET /health
GET /ready
GET /metrics
GET /docs
GET /redoc
```

## Health vs Readiness

### `/health`

Checks process liveness.

### `/ready`

Checks whether PostgreSQL is available.

Gemini is intentionally not required for readiness because parts of the system, including deterministic Product Search functionality, can operate without an active LLM connection.

---

# 🛡️ Error Handling

The API uses a safe error envelope:

```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An internal error occurred.",
    "trace_id": "..."
  }
}
```

Supported HTTP error classes include:

```text
400 Bad Request
401 Unauthorized
4
```

---

