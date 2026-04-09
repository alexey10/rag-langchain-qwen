from langchain_community.vectorstores import Chroma
from app.config import CHROMA_PATH

def create_vectorstore(documents, embedding):
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )

def load_vectorstore(embedding):
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )
