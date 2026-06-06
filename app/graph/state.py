from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    answer: str
    validation: str
    validation_reason: str
    retry_count: int
    selected_docs: list[str]
