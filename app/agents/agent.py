from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
import threading
import time

from google.genai import types

from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tool_registry import get_tool_declarations
from app.core.config import CONVERSATION_HISTORY_LIMIT
from app.core.context import RequestContext
from app.core.gemini import GeminiClient
from app.core.security import sanitize_short_id, sanitize_tool_arguments
from app.core.observability import (
    estimate_llm_cost_usd,
    llm_purpose_scope,
    log_event,
    observe,
)
from app.tools.order_tools import OrderTools
from app.tools.customer_tools import CustomerTools
from app.tools.knowledge_tools import KnowledgeTools
from app.tools.product_tools import (
    EMPTY_PRODUCT_SEARCH_MESSAGE,
    ProductTools,
    serialize_product_search_results,
)
from app.tools.conversation_tools import ConversationTools


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class CustomerSupportAgent:

    MAX_TOOL_ROUNDS = 5

    def __init__(
        self,
        gemini_client: GeminiClient,
        order_tools: OrderTools,
        customer_tools: CustomerTools,
        knowledge_tools: KnowledgeTools,
        product_tools: ProductTools,
        conversation_tools: ConversationTools,
    ):
        self.gemini = gemini_client
        self.order_tools = order_tools
        self.customer_tools = customer_tools
        self.knowledge_tools = knowledge_tools
        self.product_tools = product_tools
        self.conversation_tools = conversation_tools
        self._session_lock = threading.Lock()

    def run(
        self,
        user_message: str,
        context: RequestContext,
    ) -> AgentResult:
        started = time.perf_counter()

        tools = [
            types.Tool(
                function_declarations=get_tool_declarations()
            )
        ]

        chat_config = {
            "system_instruction": SYSTEM_PROMPT,
            "tools": tools,
        }
        thinking = getattr(self.gemini, "thinking_config", None)
        if callable(thinking):
            chat_config["thinking_config"] = thinking()
        create_kwargs = {
            "model": self.gemini.model,
            "config": types.GenerateContentConfig(**chat_config),
        }
        history = self._history_contents(context)
        if history:
            create_kwargs["history"] = history
        chat = self.gemini.client.chats.create(**create_kwargs)

        response = self._send_model_message(
            chat,
            user_message,
            purpose="agent_reasoning",
        )

        tool_calls: list[ToolCallRecord] = []
        empty_catalog_search = False

        for _ in range(self.MAX_TOOL_ROUNDS):

            if not response.function_calls:
                break

            function_responses = []
            executed = self._run_tool_calls(
                list(response.function_calls),
                context,
            )
            for record, payload, empty_search in executed:
                tool_calls.append(record)
                empty_catalog_search = empty_catalog_search or empty_search
                function_responses.append(
                    types.Part.from_function_response(
                        name=record.name,
                        response=payload,
                    )
                )

            response = self._send_model_message(
                chat,
                function_responses,
                purpose="agent_tool_followup",
            )

        else:
            return self._finish(
                context=context,
                user_message=user_message,
                answer=(
                    "متأسفانه در بررسی اطلاعات "
                    "مشکلی پیش آمد. لطفاً دوباره تلاش کنید."
                ),
                tool_calls=tool_calls,
                started=started,
            )

        answer = response.text
        catalog_only = bool(tool_calls) and all(
            call.name == "search_products"
            for call in tool_calls
        )
        if empty_catalog_search and catalog_only:
            answer = (
                f"{EMPTY_PRODUCT_SEARCH_MESSAGE}. "
                "اگر برند، نوع کالا یا بازه قیمت دقیق‌تری بگویید "
                "بهتر راهنمایی می‌کنم."
            )

        return self._finish(
            context=context,
            user_message=user_message,
            answer=answer,
            tool_calls=tool_calls,
            started=started,
        )

    def _run_tool_calls(
        self,
        function_calls,
        context: RequestContext,
    ) -> list[tuple[ToolCallRecord, dict, bool]]:
        if len(function_calls) <= 1:
            return [
                self._execute_one_tool(function_call, context)
                for function_call in function_calls
            ]

        results: list[tuple[ToolCallRecord, dict, bool] | None] = [
            None
        ] * len(function_calls)

        def run_index(index: int):
            function_call = function_calls[index]
            name = function_call.name
            if name == "search_knowledge_base":
                results[index] = self._execute_one_tool(function_call, context)
                return
            with self._session_lock:
                results[index] = self._execute_one_tool(function_call, context)

        with ThreadPoolExecutor(max_workers=min(4, len(function_calls))) as pool:
            list(pool.map(run_index, range(len(function_calls))))
        return [item for item in results if item is not None]

    def _execute_one_tool(
        self,
        function_call,
        context: RequestContext,
    ) -> tuple[ToolCallRecord, dict, bool]:
        name = function_call.name
        args = sanitize_tool_arguments(function_call.args)
        log_event("agent_tool_selection", tool=name)
        record = ToolCallRecord(name=name, arguments=dict(args))
        tool_started = time.perf_counter()
        tool_status = "success"
        tool_error_type = None
        empty_search = False
        try:
            payload, empty_search = self._invoke_tool(name, args, context)
        except Exception as exc:
            tool_status = "failure"
            tool_error_type = type(exc).__name__
            payload = {
                "error": (
                    "The requested data "
                    "could not be retrieved."
                )
            }
        log_event(
            "tool_execution",
            tool=name,
            status=tool_status,
            latency_ms=round((time.perf_counter() - tool_started) * 1000, 3),
            error_type=tool_error_type,
        )
        return record, payload, empty_search

    def _invoke_tool(
        self,
        name: str,
        args: dict[str, Any],
        context: RequestContext,
    ) -> tuple[dict, bool]:
        empty_catalog_search = False
        if name == "get_customer_profile":

            fields = args.get("fields")

            result = (
                self.customer_tools
                .get_customer_profile(
                    context=context,
                    fields=fields,
                )
            )

            payload = (
                result.model_dump()
                if result
                else {
                    "found": False,
                    "message": (
                        "Customer profile was not found."
                    ),
                }
            )

                    # =========================
                    # ORDERS
                    # =========================

        elif name == "get_customer_order_summary":

            result = (
                self.order_tools
                .get_customer_order_summary(
                    context=context
                )
            )

            payload = result.model_dump()

        elif name == "get_latest_order":

            result = (
                self.order_tools
                .get_latest_order(
                    context=context
                )
            )

            payload = (
                result.model_dump()
                if result
                else {
                    "found": False
                }
            )

        elif name == "get_order_status":

            order_id = sanitize_short_id(
                args.get("order_id", "")
            )

            if not order_id:
                payload = {
                    "error": "order_id is required"
                }

            else:

                result = (
                    self.order_tools
                    .get_order_status(
                        context=context,
                        order_id=order_id,
                    )
                )

                payload = (
                    result.model_dump()
                    if result
                    else {
                        "found": False,
                        "message": (
                            "No order with this ID "
                            "was found for the "
                            "authenticated customer."
                        ),
                    }
                )

        elif name == "get_order_details":

            order_id = sanitize_short_id(
                args.get("order_id", "")
            )

            if not order_id:
                payload = {
                    "error": "order_id is required"
                }

            else:

                result = (
                    self.order_tools
                    .get_order_details(
                        context=context,
                        order_id=order_id,
                    )
                )

                payload = {
                    "found": bool(result),
                    "items": [
                        item.model_dump()
                        for item in result
                    ],
                }

        elif name == "get_order_history":

            limit = int(
                args.get("limit", 20)
            )

            limit = min(
                max(limit, 1),
                50,
            )

            result = (
                self.order_tools
                .get_order_history(
                    context=context,
                    limit=limit,
                )
            )

            payload = {
                "orders": [
                    order.model_dump()
                    for order in result
                ]
            }

        elif name == "search_customer_orders":

            search_term = str(
                args.get("search_term", "")
            ).strip()

            if not search_term:

                payload = {
                    "error": "search_term is required"
                }

            else:

                limit = int(
                    args.get("limit", 20)
                )

                limit = min(
                    max(limit, 1),
                    50,
                )

                result = (
                    self.order_tools
                    .search_customer_orders(
                        context=context,
                        search_term=search_term,
                        limit=limit,
                    )
                )

                payload = {
                    "results": [
                        item.model_dump()
                        for item in result
                    ]
                }

        elif name == "get_purchased_products":

            limit = int(
                args.get("limit", 50)
            )

            limit = min(
                max(limit, 1),
                100,
            )

            result = (
                self.order_tools
                .get_purchased_products(
                    context=context,
                    limit=limit,
                )
            )

            payload = {
                "products": [
                    product.model_dump()
                    for product in result
                ]
            }

        elif name == "get_product_purchase_history":

            product_query = str(
                args.get("product_query", "")
            ).strip()

            if not product_query:

                payload = {
                    "error": "product_query is required"
                }

            else:

                limit = int(
                    args.get("limit", 20)
                )

                limit = min(
                    max(limit, 1),
                    50,
                )

                result = (
                    self.order_tools
                    .get_product_purchase_history(
                        context=context,
                        product_query=product_query,
                        limit=limit,
                    )
                )

                payload = {
                    "results": [
                        item.model_dump()
                        for item in result
                    ]
                }

        elif name == "search_knowledge_base":

            query = str(
                args.get("query", "")
            ).strip()

            if not query:
                payload = {
                    "error": "query is required"
                }

            else:

                limit = int(
                    args.get("limit", 3)
                )

                limit = min(
                    max(limit, 1),
                    10,
                )

                result = (
                    self.knowledge_tools
                    .search_knowledge_base(
                        context=context,
                        query=query[:500],
                        limit=limit,
                    )
                )

                payload = {
                    "found": bool(result),
                    "results": [
                        item.model_dump()
                        for item in result
                    ],
                }

        elif name == "search_products":

            query = str(
                args.get("query", "")
            ).strip()

            if not query:
                empty_catalog_search = True
                payload = {
                    "error": "query is required"
                }

            else:

                result = (
                    self.product_tools
                    .search_products(
                        context=context,
                        query=query[:200],
                    )
                )
                empty_catalog_search = not result
                payload = (
                    serialize_product_search_results(
                        result
                    )
                )

        elif name == "get_conversation_history":

            limit = int(
                args.get("limit", 10)
            )

            limit = min(
                max(limit, 1),
                50,
            )

            result = (
                self.conversation_tools
                .get_conversation_history(
                    context=context,
                    limit=limit,
                )
            )

            payload = {
                "found": bool(result),
                "messages": [
                    item.model_dump()
                    for item in result
                ],
            }

        elif name == "search_old_conversations":

            search_term = str(
                args.get("search_term", "")
            ).strip()

            if not search_term:
                payload = {
                    "error": "search_term is required"
                }

            else:

                limit = int(
                    args.get("limit", 10)
                )

                limit = min(
                    max(limit, 1),
                    50,
                )

                result = (
                    self.conversation_tools
                    .search_old_conversations(
                        context=context,
                        search_term=search_term[:200],
                        limit=limit,
                    )
                )

                payload = {
                    "found": bool(result),
                    "messages": [
                        item.model_dump()
                        for item in result
                    ],
                }

        else:

            log_event("agent_tool_selection_error", tool=name)
            payload = {
                "error": "Unknown tool"
            }
        return payload, empty_catalog_search

    def _history_contents(self, context: RequestContext) -> list:
        if CONVERSATION_HISTORY_LIMIT <= 0:
            return []
        try:
            messages = self.conversation_tools.get_conversation_history(
                context=context,
                limit=CONVERSATION_HISTORY_LIMIT,
            ) or []
        except Exception as exc:
            log_event(
                "conversation_history_failed",
                error_type=type(exc).__name__,
                status="failure",
            )
            return []

        contents = []
        for message in messages:
            role = str(getattr(message, "role", "") or "").strip()
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            contents.append(
                types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part(text=content[:2000])],
                )
            )
        return contents

    def _send_model_message(self, chat, message, purpose: str):
        with llm_purpose_scope(purpose):
            with observe(
                "llm_call",
                model=self.gemini.model,
                purpose=purpose,
            ) as extras:
                extras["timeout"] = False
                try:
                    response = chat.send_message(message)
                except Exception as exc:
                    extras["timeout"] = _is_timeout(exc)
                    raise
                extras.update(_usage_fields(response, self.gemini.model))
                return response

    def _finish(
        self,
        context: RequestContext,
        user_message: str,
        answer: str,
        tool_calls: list[ToolCallRecord],
        started: float | None = None,
    ) -> AgentResult:
        try:
            self.conversation_tools.save_turn(
                context=context,
                user_message=user_message,
                assistant_message=answer,
            )
        except Exception as exc:
            log_event(
                "conversation_save_failed",
                error_type=type(exc).__name__,
                status="failure",
            )

        if started is not None:
            log_event(
                "agent_completed",
                status="success",
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
                tool_count=len(tool_calls),
                tools=[call.name for call in tool_calls],
            )

        return AgentResult(
            answer=answer,
            tool_calls=tool_calls,
        )


def _is_timeout(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "deadline" in text


def _usage_fields(response: Any, model: str) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    fields: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    cost = estimate_llm_cost_usd(input_tokens, output_tokens, model)
    if cost is not None:
        fields["estimated_cost_usd"] = cost
    return fields