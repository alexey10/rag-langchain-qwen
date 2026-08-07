from langchain_ollama import OllamaLLM
from app.config import (
    LLM_KEEP_ALIVE,
    LLM_MODEL,
    LLM_NUM_PREDICT,
    LLM_REASONING,
)

def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1,
        num_predict=LLM_NUM_PREDICT,
        keep_alive=LLM_KEEP_ALIVE,
        reasoning=LLM_REASONING
    )


def warm_llm():
    llm = get_llm()

    return llm.invoke(
        "Reply with OK only."
    )
