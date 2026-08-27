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

        unavailable = []

        for model_id, config in MODELS.items():
            runtime_model = config["model"]

            if runtime_model not in available_models:
                unavailable.append(runtime_model)

        if unavailable:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "One or more configured models are unavailable",
                    "unavailable_models": unavailable,
                },
            )

        return {
            "status": "ready",
            "models": list(MODELS.keys()),
        }

    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "Ollama is unavailable",
            },
        )
