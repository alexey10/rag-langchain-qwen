from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from app.api.auth import verify_api_key
from app.api.schemas import ChatRequest, ChatResponse, ModelsResponse, ModelInfo
from app.gateway.router import get_provider
from app.gateway.models import MODELS

security = HTTPBearer()

router = APIRouter(prefix="/v1")


@router.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
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


@router.get("/models", response_model=ModelsResponse, dependencies=[Depends(verify_api_key)])
def list_models():
    return {
        "models": [
            {"id": model_id, **meta}
            for model_id, meta in MODELS.items()
        ]
    }
