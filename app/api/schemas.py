from pydantic import BaseModel
from typing import List


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
