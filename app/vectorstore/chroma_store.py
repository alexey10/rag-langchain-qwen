from langchain_chroma import Chroma
from app.config import CHROMA_PATH


def create_vectorstore(documents, embedding):
    """
    Create and persist a new Chroma vector store from documents.
    """

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )

    # NOTE:
    # No need for vectorstore.persist() in modern Chroma
    # Persistence is handled automatically via persist_directory

    return vectorstore


def load_vectorstore(embedding):
    """
    Load existing Chroma vector store from disk.
    """

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )
