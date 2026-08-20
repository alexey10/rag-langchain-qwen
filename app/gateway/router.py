from app.gateway.providers.ollama import OllamaProvider


def get_provider(model):
    normalized_model = model.lower()

    if normalized_model in {
        "qwen",
        "qwen3",
        "ollama",
    }:
        return OllamaProvider()

    raise ValueError(
        f"Unsupported model provider: {model}"
    )
