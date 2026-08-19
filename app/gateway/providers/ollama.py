from app.llm.qwen_llm import get_llm

from .base import LLMProvider


class OllamaProvider(LLMProvider):

    def chat(
        self,
        messages,
        model=None,
        **kwargs
    ):
        llm = get_llm()

        prompt = "\n".join(
            f"{message.role}: {message.content}"
            for message in messages
        )

        return llm.invoke(prompt)
