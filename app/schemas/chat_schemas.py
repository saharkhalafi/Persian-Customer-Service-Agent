from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "کرم ضد چروک پرایم دارید؟"},
                {"message": "سفارش آخرم چه وضعیتی دارد؟"},
            ]
        }
    )

    message: str = Field(
        min_length=1,
        max_length=4000,
        description="Customer message in Persian or mixed language.",
        examples=["کرم ضد چروک پرایم دارید؟"],
    )


class ToolCallInfo(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "چند کرم ضد چروک پرایم پیدا کردم. می‌توانید از بین نتایج انتخاب کنید.",
                    "tool_calls": [
                        {
                            "name": "search_products",
                            "arguments": {"query": "کرم ضد چروک پرایم"},
                        }
                    ],
                    "request_id": "4f2c8a1b9e0d47c0a1b2c3d4e5f60789",
                    "trace_id": "4f2c8a1b9e0d47c0a1b2c3d4e5f60789",
                }
            ]
        }
    )

    answer: str
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    request_id: str = ""
    trace_id: str = ""


class ErrorBody(BaseModel):
    code: str
    message: str
    trace_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
