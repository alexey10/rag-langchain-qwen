from fastapi import APIRouter
from gateway.router import get_provider

router = APIRouter(prefix="/v1")


@router.post("/chat")
def chat(request):
    provider = get_provider(request.model)

    response = provider.chat(
        messages=request.messages,
        model=request.model
    )

    return response
