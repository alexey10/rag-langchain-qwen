from app.llm.qwen_llm import get_llm

from .base import LLMProvider


class OllamaProvider(LLMProvider):

    def __init__(self, model):
        self.model = model

    def chat(
        self,
        messages,
        model=None,
        **kwargs
    ):
        llm = get_llm(model or self.model)

        prompt = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        return llm.invoke(prompt)
