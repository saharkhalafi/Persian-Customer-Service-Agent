from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.feedback import router as feedback_router
from app.core.config import CORS_ALLOWED_ORIGINS
from app.core.database import check_database
from app.core.errors import error_payload, register_exception_handlers
from app.core.logging import configure_logging
from app.core.metrics import registry
from app.core.middleware import RequestContextMiddleware
from app.core.observability import get_trace_id

API_DESCRIPTION = """
Persian e-commerce customer-support agent.

This API runs the **real** Agent pipeline (Gemini tool calling → tools →
services → repositories). It is not a demo stub.

## Authentication / context

Send the authenticated customer on every `/api/v1/chat` and `/api/v1/feedback` request:

- `X-Customer-ID`: required backend/session customer identifier
- `X-Request-ID`: optional; propagated when present and well-formed
- `X-Trace-ID`: optional; propagated when present and well-formed
- `traceparent`: optional W3C header; used when `X-Trace-ID` is absent

Never put `customer_id` in the JSON body. The model never receives it as a
tool argument.

## Local testing

1. Set `DATABASE_URL` and `GEMINI_API_KEY` (see `.env.example`).
2. Apply schema: `alembic upgrade head` (never created implicitly at startup).
3. Start (dev): `uvicorn app.main:app --reload --port 8000`
4. Open `/docs`, `/health`, `/ready`, and `/metrics`.
Production: `ENVIRONMENT=production` and `uvicorn` without `--reload` (see Docker Compose).
"""


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Modiseh AI Customer Support",
        version="1.0.0",
        description=API_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Modiseh Support Agent"},
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-Customer-ID",
            "X-Request-ID",
            "X-Trace-ID",
            "traceparent",
        ],
        expose_headers=["X-Request-ID", "X-Trace-ID", "Retry-After"],
    )
    register_exception_handlers(app)
    app.include_router(chat_router)
    app.include_router(feedback_router)

    @app.get("/health", tags=["ops"], summary="Liveness probe")
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"], summary="Readiness probe")
    def ready():
        database_ok = check_database()
        payload = {
            "status": "ready" if database_ok else "not_ready",
            "checks": {
                "database": "ok" if database_ok else "unavailable",
            },
            "trace_id": get_trace_id() or None,
        }
        if database_ok:
            return payload
        envelope = error_payload(503, "DEPENDENCY_UNAVAILABLE")
        envelope.update(payload)
        return JSONResponse(status_code=503, content=envelope)

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus-compatible in-process metrics",
        response_class=PlainTextResponse,
    )
    def metrics():
        return PlainTextResponse(
            registry.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        components = schema.setdefault("components", {})
        schemes = components.setdefault("securitySchemes", {})
        schemes["CustomerId"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-Customer-ID",
            "description": "Authenticated customer id from the backend/session.",
        }
        schemes["RequestId"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-Request-ID",
        }
        schemes["TraceId"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-Trace-ID",
        }
        for path, methods in schema.get("paths", {}).items():
            if path not in {"/api/v1/chat", "/api/v1/feedback"}:
                continue
            for spec in methods.values():
                if isinstance(spec, dict):
                    spec["security"] = [{"CustomerId": []}]
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi
    return app


app = create_app()
