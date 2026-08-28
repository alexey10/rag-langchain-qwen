from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends, Request
from app.api.auth import verify_api_key
from app.api.schemas import ChatRequest, ChatResponse, ModelsResponse, ModelInfo
from app.api.ratelimit import limiter
from app.gateway.router import get_provider
from app.gateway.models import MODELS

router = APIRouter(prefix="/v1")


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("100/day")
def chat(request: Request, body: ChatRequest):
    try:
        provider = get_provider(body.model)
        response = provider.chat(
            messages=body.messages,
        )
        return {
            "id": f"chatcmpl-{uuid4()}",
            "model": body.model,
            "content": response,
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )


@router.get("/models", response_model=ModelsResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("1000/day")
def list_models(request: Request):
    return {
        "models": [
            {"id": model_id, **meta}
            for model_id, meta in MODELS.items()
        ]
    }
