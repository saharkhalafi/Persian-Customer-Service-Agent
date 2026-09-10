from dataclasses import dataclass, field
from typing import Any

from google.genai import types

from app.agents.prompts import SYSTEM_PROMPT
from app.agents.tool_registry import get_tool_declarations
from app.core.context import RequestContext
from app.core.gemini import GeminiClient
from app.tools.order_tools import OrderTools
from app.tools.customer_tools import CustomerTools
from app.tools.knowledge_tools import KnowledgeTools
from app.tools.product_tools import ProductTools
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

    def run(
        self,
        user_message: str,
        context: RequestContext,
    ) -> AgentResult:

        tools = [
            types.Tool(
                function_declarations=get_tool_declarations()
            )
        ]

        chat = self.gemini.client.chats.create(
            model=self.gemini.model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=tools,
            ),
        )

        response = chat.send_message(user_message)

        tool_calls: list[ToolCallRecord] = []

        for _ in range(self.MAX_TOOL_ROUNDS):

            if not response.function_calls:
                break

            function_responses = []

            for function_call in response.function_calls:

                name = function_call.name
                args = function_call.args or {}

                tool_calls.append(
                    ToolCallRecord(
                        name=name,
                        arguments=dict(args),
                    )
                )

                try:

                    # =========================
                    # CUSTOMER
                    # =========================

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

                        order_id = str(
                            args.get("order_id", "")
                        ).strip()

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

                        order_id = str(
                            args.get("order_id", "")
                        ).strip()

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
                            payload = {
                                "error": "query is required"
                            }

                        else:

                            limit = int(
                                args.get("limit", 10)
                            )

                            limit = min(
                                max(limit, 1),
                                20,
                            )

                            result = (
                                self.product_tools
                                .search_products(
                                    context=context,
                                    query=query[:200],
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

                        payload = {
                            "error": "Unknown tool"
                        }

                except Exception:

                    payload = {
                        "error": (
                            "The requested data "
                            "could not be retrieved."
                        )
                    }

                function_responses.append(
                    types.Part.from_function_response(
                        name=name,
                        response=payload,
                    )
                )

            response = chat.send_message(
                function_responses
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
            )

        return self._finish(
            context=context,
            user_message=user_message,
            answer=response.text,
            tool_calls=tool_calls,
        )

    def _finish(
        self,
        context: RequestContext,
        user_message: str,
        answer: str,
        tool_calls: list[ToolCallRecord],
    ) -> AgentResult:
        try:
            self.conversation_tools.save_turn(
                context=context,
                user_message=user_message,
                assistant_message=answer,
            )
        except Exception:
            pass

        return AgentResult(
            answer=answer,
            tool_calls=tool_calls,
        )