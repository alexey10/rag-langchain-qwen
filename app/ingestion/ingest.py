from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.embeddings.embedding import get_embedding_model
from app.vectorstore.chroma_store import create_vectorstore
from app.config import DATA_PATH

def run_ingestion():
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)

    embedding = get_embedding_model()
    vectorstore = create_vectorstore(chunks, embedding)

    vectorstore.persist()
    print("✅ Ingestion complete")
