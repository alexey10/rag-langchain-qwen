from langchain_ollama import OllamaLLM

from app.config import (
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_NUM_PREDICT,
    LLM_REASONING,
    OLLAMA_HOST,
)


def get_llm(model=None):
    model = model or LLM_MODEL

    return OllamaLLM(
        model=model,
        base_url=OLLAMA_HOST,
        temperature=0.1,
        num_predict=LLM_NUM_PREDICT,
        keep_alive=LLM_KEEP_ALIVE,
        reasoning=LLM_REASONING,
    )


def warm_llm():
    llm = get_llm()

    return llm.invoke(
        "Reply with OK only."
    )
