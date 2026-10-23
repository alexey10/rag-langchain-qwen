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


class KnowledgeBaseInfo(BaseModel):
    id: str
    name: str


class KnowledgeBasesResponse(BaseModel):
    knowledge_bases: List[KnowledgeBaseInfo]


class DocumentInfo(BaseModel):
    id: str
    knowledge_base_id: str
    filename: str
    status: str
    size_bytes: int
    chunk_count: int
    created_at: str


class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
