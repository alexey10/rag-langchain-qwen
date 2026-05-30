

from langsmith import traceable

from app.retrieval.retriever import get_retriever
from app.chains.rag_chain import build_rag_chain

retriever = get_retriever()
qa_chain = build_rag_chain(retriever)

#Retrieval Node

@traceable(name="retrieve")
def retrieve(state):
    docs = retriever.get_relevant_documents(
        state["question"]
    )

    return {
        "documents": docs
    }

#Generation Node

@traceable(name="generate")
def generate(state):

    docs = state["documents"]

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Answer using only the supplied context.

Context:
{context}

Question:
{state['question']}
"""

    answer = qa_chain.invoke(prompt)

    return {
        "answer": answer
    }

#Validation Node

@traceable(name="validate")
def validate(state):

    answer = state["answer"]

    if len(answer) < 20:
        validation = "RETRY"
    else:
        validation = "PASS"

    return {
        "validation": validation
    }
