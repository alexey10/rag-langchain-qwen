from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.embeddings.embedding import get_embedding_model
from app.vectorstore.chroma_store import create_vectorstore
from app.config import DATA_PATH


def run_ingestion(knowledge_base_id="default"):
    docs = load_documents(DATA_PATH)
    chunks = split_documents(docs)

    for chunk in chunks:
        chunk.metadata["knowledge_base_id"] = knowledge_base_id

    embedding = get_embedding_model()
    vectorstore = create_vectorstore(
        chunks,
        embedding,
    )

    print(f"Knowledge base: {knowledge_base_id}")
    print(f"Loaded docs: {len(docs)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content[:500])
    print("\nFirst chunk metadata:\n")
    print(chunks[0].metadata)

    print("✅ Ingestion complete")
