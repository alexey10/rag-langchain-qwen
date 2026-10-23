from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from app.ingestion.splitter import split_documents
from app.embeddings.embedding import get_embedding_model
from app.vectorstore.chroma_store import load_vectorstore


def index_pdf_document(
    file_path: str | Path,
    knowledge_base_id: str,
    document_id: str,
    original_filename: str,
) -> int:
    pages = PyPDFLoader(str(file_path)).load()

    if not pages:
        raise ValueError("PDF contains no readable pages")

    chunks = split_documents(pages)

    if not chunks:
        raise ValueError("PDF produced no indexable chunks")

    for chunk in chunks:
        chunk.metadata["knowledge_base_id"] = knowledge_base_id
        chunk.metadata["document_id"] = document_id
        chunk.metadata["original_filename"] = original_filename

    embedding = get_embedding_model()
    vectorstore = load_vectorstore(embedding)

    vectorstore.add_documents(
        documents=chunks,
        ids=[
            f"{document_id}:{index}"
            for index in range(len(chunks))
        ],
    )

    return len(chunks)


def remove_document_chunks(document_id: str) -> None:
    embedding = get_embedding_model()
    vectorstore = load_vectorstore(embedding)

    vectorstore.delete(
        where={"document_id": document_id}
    )
