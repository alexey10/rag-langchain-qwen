from langchain_community.llms import Ollama
from app.config import LLM_MODEL

def get_llm():
    return Ollama(
        model=LLM_MODEL,
        temperature=0.1
    )
