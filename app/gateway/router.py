from app.gateway.models import MODELS
from app.gateway.providers.ollama import OllamaProvider


def get_provider(model):
    normalized_model = model.lower()

    if normalized_model not in MODELS:
        raise ValueError(
            f"Unsupported model: {model}"
        )

    provider_name = MODELS[normalized_model]["provider"]

    if provider_name == "ollama":
        return OllamaProvider()

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )
