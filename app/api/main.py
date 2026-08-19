from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="deepVerified API"
)

app.include_router(router)
