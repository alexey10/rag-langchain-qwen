from app.retrieval.retriever import (
    get_retriever
)

from app.retrieval.bm25_retriever import (
    get_bm25_retriever
)


def hybrid_search(query):

    vector_docs = get_retriever().invoke(query)

    bm25_docs = get_bm25_retriever().invoke(query)

    combined = vector_docs + bm25_docs

    unique_docs = []

    seen = set()

    for doc in combined:

        text = doc.page_content

        if text not in seen:

            seen.add(text)

            unique_docs.append(doc)

    return unique_docs[:5]
