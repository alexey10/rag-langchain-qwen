from fastapi import FastAPI

from app.api.routes import router
from app.api.health import router as health_router

app = FastAPI(
    title="deepVerified API"
)

app.include_router(router)
app.include_router(health_router)
