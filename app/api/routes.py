from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer

from app.api.client_registry import (
    APIClient,
    KnowledgeBaseAccessError,
)

from app.api.auth import verify_api_key
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ModelsResponse,
    KnowledgeBasesResponse,
)
from app.gateway.router import get_provider
from app.gateway.availability import (
    ModelUnavailableError,
    OllamaUnavailableError,
    get_available_model_config,
    get_model_catalog,
)
from app.services.rag_service import RAGService

from app.api.knowledge_base_registry import (
    list_knowledge_bases,
)

security = HTTPBearer()

router = APIRouter(prefix="/v1")

rag_service = RAGService()


@router.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest,client: APIClient = Depends(verify_api_key),):
    try:
        if request.knowledge_base_id:
            content = rag_service.answer(
                messages=request.messages,
                knowledge_base_id=request.knowledge_base_id,
                model=request.model,
                client=client,
            )
        else:
            get_available_model_config(request.model)
            provider = get_provider(request.model)

            content = provider.chat(
                messages=request.messages,
            )

        return {
            "id": f"chatcmpl-{uuid4()}",
            "model": request.model,
            "content": content,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except KnowledgeBaseAccessError:
        raise HTTPException(
            status_code=404,
            detail="Knowledge base not found",
        )


@router.get(
    "/models",
    response_model=ModelsResponse,
    dependencies=[Depends(verify_api_key)],
)
def list_models():
    try:
        return {"models": get_model_catalog()}
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

@router.get(
    "/knowledge-bases",
    response_model=KnowledgeBasesResponse,
)
def list_client_knowledge_bases(
    client: APIClient = Depends(verify_api_key),
):
    active_knowledge_bases = list_knowledge_bases(
        status="active"
    )

    return {
        "knowledge_bases": [
            {
                "id": kb.id,
                "name": kb.name,
            }
            for kb in active_knowledge_bases
            if kb.id in client.allowed_knowledge_bases
        ]
    }
