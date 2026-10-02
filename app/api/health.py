from fastapi import APIRouter
from fastapi.responses import JSONResponse
import ollama
from app.gateway.models import MODELS

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok"
    }


@router.get("/ready")
def readiness():
    try:
        response = ollama.list()
        available_models = [model.model for model in response.models]

        required_model = MODELS["qwen"]["model"]

        if required_model not in available_models:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "Required model is unavailable",
                    "required_model": required_model,
                },
            )

        return {
            "status": "ready",
            "models": list(MODELS.keys()),
            "available_runtime_models": available_models,
        }

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Ollama is unavailable",
            },
        )
