from pydantic import BaseModel
from typing import List


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen"
    messages: List[ChatMessage]
    knowledge_base_id: str | None = None


class ChatResponse(BaseModel):
    id: str
    model: str
    content: str


class ModelInfo(BaseModel):
    id: str
    provider: str
    model: str
    available: bool


class ModelsResponse(BaseModel):
    models: List[ModelInfo]
