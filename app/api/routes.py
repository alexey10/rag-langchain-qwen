from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse, ModelsResponse, ModelInfo
from app.gateway.router import get_provider
from app.gateway.models import MODELS

router = APIRouter(prefix="/v1")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        provider = get_provider(request.model)

        response = provider.chat(
            messages=request.messages,
        )

        return {
            "id": f"chatcmpl-{uuid4()}",
            "model": request.model,
            "content": response,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

@router.get("/models", response_model=ModelsResponse)
def list_models():
    return {
        "models": [
            {"id": model_id, **meta}
            for model_id, meta in MODELS.items()
        ]
    }
