from fastapi import APIRouter, Depends, status

from app.agents.agent import CustomerSupportAgent
from app.api.dependencies import get_agent
from app.core.context import RequestContext
from app.core.rate_limit import enforce_chat_rate_limit
from app.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ToolCallInfo,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)

CHAT_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Validation error"},
    401: {"model": ErrorResponse, "description": "Missing X-Customer-ID"},
    403: {"model": ErrorResponse, "description": "Not authorized"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
    429: {"model": ErrorResponse, "description": "Rate limited"},
    500: {"model": ErrorResponse, "description": "Internal error"},
    503: {"model": ErrorResponse, "description": "Dependency unavailable"},
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a customer message to the support agent",
    description=(
        "Executes the production Agent: Gemini function calling, existing tools, "
        "Product Search, orders, knowledge, and conversation memory. "
        "Requires `X-Customer-ID`. Optional `X-Request-ID` / `X-Trace-ID` are "
        "propagated through logs and the response."
    ),
    responses=CHAT_RESPONSES,
)
def chat(
    request: ChatRequest,
    context: RequestContext = Depends(enforce_chat_rate_limit),
    agent: CustomerSupportAgent = Depends(get_agent),
):
    result = agent.run(
        user_message=request.message,
        context=context,
    )

    return ChatResponse(
        answer=result.answer,
        tool_calls=[
            ToolCallInfo(
                name=call.name,
                arguments=call.arguments,
            )
            for call in result.tool_calls
        ],
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
