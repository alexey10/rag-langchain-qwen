from fastapi import FastAPI
from fastapi.security import HTTPBearer
from app.api.routes import router
from app.api.health import router as health_router

security = HTTPBearer()

app = FastAPI(
    title="deepVerified API",
    version="1.0.0",
    description="AI API layer for RAG, Agents and LLMs",
    openapi_tags=[{"name": "v1"}],
)

app.include_router(router)
app.include_router(health_router)

# Register Bearer auth scheme in OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
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
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi
