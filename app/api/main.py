from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.api.routes import router
from app.api.health import router as health_router
from app.api.middleware import LoggingMiddleware
from app.api.ratelimit import limiter

app = FastAPI(
    title="deepVerified API",
    version="1.0.0",
    description="AI API layer for RAG, Agents and LLMs",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(LoggingMiddleware)
app.include_router(router)
app.include_router(health_router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
        }
    }

    for path, path_item in schema.get("paths", {}).items():
        if path in ("/health", "/ready"):
            continue
        for operation in path_item.values():
            operation["security"] = [{"BearerAuth": []}]

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
