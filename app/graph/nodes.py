
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

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = f"""
You are evaluating a RAG response.

Question:
{state["question"]}

Retrieved Context:
{context}

Answer:
{state["answer"]}

Determine:

1. Is the answer supported by the context?
2. Is it relevant to the question?
3. Is it reasonably complete?

Respond ONLY with:

PASS

or

RETRY
"""

    verdict = llm.invoke(prompt).strip()

    if "PASS" in verdict.upper():
        return {
            "validation": "PASS",
            "validation_reason": verdict,
        }

    return {
        "validation": "RETRY",
        "validation_reason": verdict,
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
