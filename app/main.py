from app.ingestion.ingest import run_ingestion
from app.retrieval.retriever import get_retriever
from app.chains.rag_chain import build_rag_chain


def main():
    # Step 1: Run once (uncomment for first run)
    # run_ingestion()

    retriever = get_retriever()
    qa_chain = build_rag_chain(retriever)

    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = qa_chain(query)

        print("\nAnswer:\n", result["result"])


if __name__ == "__main__":
    main()
