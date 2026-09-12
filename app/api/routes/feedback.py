from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_feedback_repository
from app.core.auth import get_request_context
from app.core.context import RequestContext
from app.core.observability import log_event
from app.core.security import sanitize_short_id
from app.repositories.feedback_repository import FeedbackRepository
from app.schemas.chat_schemas import ErrorResponse
from app.schemas.feedback_schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["feedback"],
)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Record thumbs-up / thumbs-down on a chat turn",
    description=(
        "Minimal feedback write. Identity comes from `X-Customer-ID` only. "
        "`conversation_id` and `message_id` are optional client references."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        401: {"model": ErrorResponse, "description": "Missing X-Customer-ID"},
        500: {"model": ErrorResponse, "description": "Internal error"},
        503: {"model": ErrorResponse, "description": "Dependency unavailable"},
    },
)
def submit_feedback(
    request: FeedbackRequest,
    context: RequestContext = Depends(get_request_context),
    repository: FeedbackRepository = Depends(get_feedback_repository),
):
    conversation_id = sanitize_short_id(request.conversation_id) or None
    message_id = sanitize_short_id(request.message_id) or None
    repository.add(
        customer_id=context.customer_id,
        rating=request.rating,
        conversation_id=conversation_id,
        message_id=message_id,
    )
    log_event(
        "feedback_recorded",
        rating=request.rating,
        has_conversation_id=bool(conversation_id),
        has_message_id=bool(message_id),
    )
    return FeedbackResponse(
        accepted=True,
        request_id=context.request_id,
        trace_id=context.trace_id,
    )
