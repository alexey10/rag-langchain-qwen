from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.api.schemas import ChatRequest, ChatResponse
from app.gateway.router import get_provider

router = APIRouter(prefix="/v1")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:
        provider = get_provider(request.model)

        response = provider.chat(
            messages=request.messages,
            model=request.model,
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
