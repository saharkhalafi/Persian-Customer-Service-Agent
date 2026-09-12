from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import ApplicationError
from app.core.observability import get_trace_id, log_event

SAFE_MESSAGES = {
    400: "درخواست نامعتبر است.",
    401: "احراز هویت مشتری الزامی است.",
    403: "دسترسی به این درخواست مجاز نیست.",
    404: "مورد درخواستی پیدا نشد.",
    429: "تعداد درخواست‌ها بیش از حد مجاز است. لطفاً کمی بعد تلاش کنید.",
    500: "در حال حاضر امکان پردازش درخواست وجود ندارد.",
    503: "سرویس موقتاً در دسترس نیست. لطفاً بعداً تلاش کنید.",
}

STATUS_TO_CODE = {
    400: "VALIDATION_ERROR",
    401: "AUTHENTICATION_ERROR",
    403: "AUTHORIZATION_ERROR",
    404: "NOT_FOUND",
    429: "RATE_LIMIT",
    500: "INTERNAL_ERROR",
    503: "DEPENDENCY_UNAVAILABLE",
}

_DEPENDENCY_MARKERS = (
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection refused",
    "connection reset",
    "service unavailable",
)


def _trace_from_request(request: Request | None) -> str | None:
    if request is not None:
        state_id = getattr(request.state, "trace_id", None)
        if state_id:
            return str(state_id)
    return get_trace_id() or None


def error_payload(
    status_code: int,
    error_code: str | None = None,
    request: Request | None = None,
) -> dict:
    return {
        "error": {
            "code": error_code or STATUS_TO_CODE.get(status_code, "INTERNAL_ERROR"),
            "message": SAFE_MESSAGES.get(status_code, SAFE_MESSAGES[500]),
            "trace_id": _trace_from_request(request),
        }
    }


def _json_error(
    status_code: int,
    error_code: str | None = None,
    request: Request | None = None,
) -> JSONResponse:
    headers = {}
    if status_code == 429:
        headers["Retry-After"] = "60"
    return JSONResponse(
        status_code=status_code,
        content=error_payload(status_code, error_code, request=request),
        headers=headers,
    )


def _http_status_from_exception(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, ApplicationError):
        return exc.status_code, exc.error_code
    if isinstance(exc, (OperationalError, TimeoutError)):
        return 503, "DEPENDENCY_UNAVAILABLE"
    if isinstance(exc, SQLAlchemyError):
        return 503, "DEPENDENCY_UNAVAILABLE"
    text = str(exc).lower()
    if any(marker in text for marker in _DEPENDENCY_MARKERS):
        return 503, "DEPENDENCY_UNAVAILABLE"
    return 500, "INTERNAL_ERROR"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def application_error_handler(
        request: Request,
        exc: ApplicationError,
    ) -> JSONResponse:
        _ = request
        log_event(
            "request_error",
            error_code=exc.error_code,
            error_type=type(exc).__name__,
            status=exc.status_code,
        )
        return _json_error(exc.status_code, exc.error_code, request=request)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        _ = exc
        log_event(
            "request_error",
            error_code="VALIDATION_ERROR",
            error_type="RequestValidationError",
            status=400,
        )
        return _json_error(400, "VALIDATION_ERROR", request=request)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        status = exc.status_code
        if status not in SAFE_MESSAGES:
            status = 500
        log_event(
            "request_error",
            error_code=STATUS_TO_CODE.get(status, "INTERNAL_ERROR"),
            error_type="HTTPException",
            status=status,
        )
        return _json_error(status, request=request)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        status, code = _http_status_from_exception(exc)
        log_event(
            "request_error",
            error_code=code,
            error_type=type(exc).__name__,
            status=status,
        )
        return _json_error(status, code, request=request)
