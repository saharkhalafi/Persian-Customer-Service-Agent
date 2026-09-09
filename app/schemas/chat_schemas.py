from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=4000,
    )


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(
        default_factory=dict
    )


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list
    )