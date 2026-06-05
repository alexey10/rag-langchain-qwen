from langchain_community.retrievers import BM25Retriever

from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.config import DATA_PATH


def get_bm25_retriever():

    docs = load_documents(DATA_PATH)

    chunks = split_documents(docs)

    retriever = BM25Retriever.from_documents(
        chunks
    )

    retriever.k = 3

    return retriever
