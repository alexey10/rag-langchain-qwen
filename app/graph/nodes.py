
from langsmith import traceable

from app.retrieval.retriever import get_retriever
from app.llm.qwen_llm import get_llm

retriever = get_retriever()
llm = get_llm()


#Retrieval Node

def retrieve(state):

    query = state.get(
    "rewritten_question",
    state["question"]
	)

    docs = retriever.invoke(query)

    return {
        "documents": docs
    }

#Generation Node

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

def validate(state):

    answer = state["answer"]

    if len(answer.strip()) < 20:
        return {
            "validation": "RETRY"
        }

    return {
        "validation": "PASS"
    }

#Rewrite Query
def rewrite_query(state):

    prompt = f"""
Rewrite the following user question to improve
document retrieval.

Return only the rewritten query.

Question:
{state["question"]}
"""

    rewritten = llm.invoke(prompt)

    return {
        "rewritten_question": rewritten.strip()
    }
