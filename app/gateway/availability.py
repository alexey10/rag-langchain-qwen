import ollama

from app.config import OLLAMA_HOST
from app.gateway.models import MODELS


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama service cannot be reached."""


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model has not been pulled into Ollama."""


def get_available_runtime_models() -> set[str]:
    try:
        response = ollama.Client(host=OLLAMA_HOST).list()
    except Exception as exc:
        raise OllamaUnavailableError("Ollama is unavailable") from exc

    return {model.model for model in response.models}


def get_model_catalog() -> list[dict]:
    available_models = get_available_runtime_models()

    return [
        {
            "id": model_id,
            **model_config,
            "available": model_config["model"] in available_models,
        }
        for model_id, model_config in MODELS.items()
    ]


def get_available_model_config(model_id: str) -> dict:
    normalized_model_id = model_id.lower()

    if normalized_model_id not in MODELS:
        raise ValueError(f"Unsupported model: {model_id}")

    model_config = MODELS[normalized_model_id]
    available_models = get_available_runtime_models()
    runtime_model = model_config["model"]

    if runtime_model not in available_models:
        raise ModelUnavailableError(
            f"Model '{runtime_model}' is not installed in Ollama. "
            f"Run: ollama pull {runtime_model}"
        )

    return model_config
