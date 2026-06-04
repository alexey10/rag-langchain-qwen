from app.vectorstore.chroma_store import load_vectorstore
from app.embeddings.embedding import get_embedding_model
from app.config import TOP_K

def get_retriever():
    embedding = get_embedding_model()
    vectorstore = load_vectorstore(embedding)

    try:
        print(
            f"Vector count: {vectorstore._collection.count()}"
        )
    except Exception as e:
        print(f"Count failed: {e}")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
