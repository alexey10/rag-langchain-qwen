

from langsmith import traceable

from app.retrieval.retriever import get_retriever
from app.llm.qwen_llm import get_llm

retriever = get_retriever()
llm = get_llm()


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

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = f"""
Answer using only the supplied context.

Context:
{context}

Question:
{state["question"]}
"""

    answer = llm.invoke(prompt)

    return {
        "answer": answer
    }


#Validation Node

@traceable(name="validate")
def validate(state):

    answer = state["answer"]

    if len(answer.strip()) < 20:
        return {
            "validation": "RETRY"
        }

    return {
        "validation": "PASS"
    }
