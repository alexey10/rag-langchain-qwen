from pydantic import BaseModel
from typing import List
from typing import Dict, Any

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen"
    messages: List[ChatMessage]


class ChatResponse(BaseModel):
    id: str
    model: str
    content: str

class ModelInfo(BaseModel):
    id: str
    provider: str
    model: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
