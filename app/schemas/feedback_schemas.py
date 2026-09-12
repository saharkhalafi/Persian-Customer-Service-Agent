from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "conversation_id": "conv-123",
                    "message_id": "msg-456",
                    "rating": "positive",
                }
            ]
        }
    )

    conversation_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional client conversation identifier.",
    )
    message_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional client message identifier.",
    )
    rating: Literal["positive", "negative"]


class FeedbackResponse(BaseModel):
    accepted: bool = True
    request_id: str = ""
    trace_id: str = ""
