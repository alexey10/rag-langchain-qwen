from functools import lru_cache

from app.vectorstore.chroma_store import load_vectorstore
from app.embeddings.embedding import get_embedding_model
from app.config import TOP_K


@lru_cache(maxsize=32)
def get_retriever(knowledge_base_id="default"):
    embedding = get_embedding_model()
    vectorstore = load_vectorstore(embedding)

    try:
        print(
            f"Vector count: "
            f"{vectorstore._collection.count()}"
        )
    except Exception as e:
        print(f"Count failed: {e}")

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": TOP_K,
            "filter": {
                "knowledge_base_id": knowledge_base_id
            },
        },
    )
