from fastapi import APIRouter, Depends

from app.agents.agent import CustomerSupportAgent
from app.api.dependencies import get_agent
from app.core.auth import get_request_context
from app.core.context import RequestContext
from app.schemas.chat_schemas import (
    ChatRequest,
    ChatResponse,
    ToolCallInfo,
)

router = APIRouter(
    prefix="/api/v1",
    tags=["chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    context: RequestContext = Depends(
        get_request_context
    ),
    agent: CustomerSupportAgent = Depends(
        get_agent
    ),
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
    )