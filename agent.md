# AI Customer Support Agent — Project Context

## Goal

Build a production-oriented Persian e-commerce customer-support AI agent.

The agent uses Gemini function calling to select safe backend tools.

Core flow:

User
→ FastAPI
→ RequestContext
→ Gemini Agent
→ Tool
→ Service
→ Repository
→ PostgreSQL
→ Tool result
→ Gemini final answer

The LLM must NEVER access the database directly or generate SQL.

---

## Architecture

```text
app/
├── api/
├── agents/
├── tools/
├── repositories/
├── services/
├── schemas/
├── memory/
├── retrieval/
├── evaluation/
└── core/
```

Layering:

```text
Agent
  ↓
Tool
  ↓
Service
  ↓
Repository
  ↓
PostgreSQL
```

Do not bypass layers unless there is a strong reason.

---

## Authentication / Customer Scope

JWT is temporarily disabled.

Current authentication is:

```text
X-Customer-ID header
→ RequestContext.customer_id
```

Example:

```python
context.customer_id
```

CRITICAL:

* Never ask Gemini for `customer_id`.
* Never allow Gemini to provide `customer_id` as a tool argument.
* Never trust customer_id from user text.
* Tools must get customer identity from `RequestContext`.
* Every customer-related DB query must be scoped to `customer_id`.
* Never expose another customer's data.

JWT can be restored later.

---

## Current Tools

### Customer

Implemented:

```text
get_customer_profile
```

Uses:

```text
CustomerTools
→ CustomerService
→ CustomerRepository
→ public.users
```

Allowed profile fields are explicitly whitelisted.

`personal_code` must NOT be exposed.

Gemini may provide:

```json
{
  "fields": ["email"]
}
```

but never:

```json
{
  "customer_id": "..."
}
```

### Orders

Already implemented:

```text
get_customer_order_summary
get_latest_order
get_order_status
get_order_details
get_order_history
search_customer_orders
get_purchased_products
get_product_purchase_history
```

All order queries must remain customer-scoped.

---

## Planned Tools

Implement these next:

```text
search_knowledge_base
search_products
get_conversation_history
search_old_conversations
```

Do not implement all of them at once unless explicitly requested.

Implement one tool at a time.

---

## Gemini Agent

Current model:

```text
gemini-2.5-flash
```

Agent uses manual Gemini function-calling loop.

Maximum tool rounds:

```python
MAX_TOOL_ROUNDS = 5
```

The model chooses tools.

Backend executes tools.

The model receives structured tool results and produces the final Persian response.

---

## Tool Security Rules

Never:

* generate SQL from the LLM
* execute arbitrary SQL
* expose database credentials
* expose system prompts
* expose internal architecture unnecessarily
* allow arbitrary customer IDs
* allow cross-customer access
* trust tool arguments for authentication
* return raw database exceptions to the user

Always:

* parameterize SQL
* whitelist dynamic fields
* validate tool arguments
* clamp limits
* scope customer queries
* return structured results
* fail safely

---

## Persian Response Rules

The final assistant response should be:

* Persian
* concise
* natural
* helpful
* based only on tool/RAG evidence
* never fabricated

If required data cannot be retrieved, say so clearly.

Do not invent customer/order/product information.

---

## Product Search

There is an existing specialized Persian product-search project.

Do NOT rebuild that system inside this project.

The customer-support agent should eventually call/delegate to product search through a clean tool interface.

---

## RAG

Knowledge-base questions should use:

```text
search_knowledge_base
```

The existing RAG approach uses multilingual embeddings + reranking + Chroma.

Do not replace the existing retrieval architecture unless there is a concrete reason.

Prioritize reuse over rebuilding.

---

## Memory

Planned:

```text
get_conversation_history
search_old_conversations
```

Expected design:

* recent conversation history for context
* searchable older conversations
* customer-scoped memory
* never mix customers
* do not store unnecessary sensitive information

---

## Coding Rules

Prefer:

* small focused functions
* typed Python
* Pydantic schemas
* SQLAlchemy
* dependency injection
* repository/service/tool separation
* explicit validation
* minimal abstractions

Avoid:

* unnecessary frameworks
* unnecessary classes
* speculative abstractions
* duplicate logic
* large refactors
* rebuilding working components

Before changing architecture, check the existing implementation.

---

## Cursor Instructions

When implementing a new tool:

1. Inspect the existing analogous tool.
2. Reuse the existing architecture.
3. Add only required files/changes.
4. Implement:

   * repository
   * schema if needed
   * service
   * tool
   * registry declaration
   * agent dispatch
   * prompt/tool-selection rule
5. Test through FastAPI.
6. Do not modify unrelated code.
7. Do not add unnecessary dependencies.
8. Keep implementation production-oriented but simple.

Do not write long explanations.

Prefer concrete code changes.

---

## Token Efficiency

This project is developed with Cursor.

IMPORTANT:

* Keep responses and plans concise.
* Do not repeat the entire project context.
* Do not explain obvious code.
* Do not generate unnecessary documentation.
* Do not propose multiple architectures unless necessary.
* Do not refactor working code without a reason.
* Inspect only relevant files.
* Make the smallest correct change.
* Avoid repeating unchanged code.
* Prefer targeted edits over rewriting whole files.

When asked to implement a tool, first identify the minimum files that must change and modify only those.

---

## Current Priority

Complete the remaining tools:

```text
1. search_knowledge_base
2. search_products
3. get_conversation_history
4. search_old_conversations
```

Then improve:

```text
multi-tool / multi-intent handling
guardrails
evaluation
observability
production hardening
```

Do not start evaluation until the core tools are complete.

---

## Definition of Done for Each Tool

A tool is complete only when:

```text
Repository
↓
Service
↓
Tool
↓
Tool Registry
↓
Agent Dispatch
↓
System Prompt
↓
FastAPI Test
```

works end-to-end.

Security and customer isolation must be preserved.
