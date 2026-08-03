from langchain_ollama import OllamaLLM
from app.config import LLM_MODEL

def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1
    )


def warm_llm():
    llm = get_llm()

    return llm.invoke(
        "Reply with OK only."
    )
