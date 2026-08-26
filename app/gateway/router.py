from app.gateway.models import MODELS
from app.gateway.providers.ollama import OllamaProvider


def get_provider(model):
    normalized_model = model.lower()

    if normalized_model not in MODELS:
        raise ValueError(
            f"Unsupported model: {model}"
        )

    model_config = MODELS[normalized_model]

    provider_name = model_config["provider"]
    runtime_model = model_config["model"]

    if provider_name == "ollama":
        return OllamaProvider(runtime_model)

    raise ValueError(
        f"Unsupported provider: {provider_name}"
    )
