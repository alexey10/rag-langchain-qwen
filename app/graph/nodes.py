
from langsmith import traceable

from app.retrieval.retriever import get_retriever
from app.llm.qwen_llm import get_llm

retriever = get_retriever()
llm = get_llm()

from app.prompts.rewrite_prompt import (
    get_rewrite_prompt
)

from app.prompts.generation_prompt import (
    get_generation_prompt
)

from app.prompts.validation_prompt import (
    get_validation_prompt
)

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

    prompt = get_generation_prompt(
        context,
        state["question"]
    )

    answer = llm.invoke(prompt)

    return {
        "answer": answer
    }

#Validation Node

def validate(state):

    prompt = get_validation_prompt(
        state["question"],
        state["answer"]
    )

    result = llm.invoke(prompt)

    return {
        "validation": result.strip()
    }

#Rewrite Query

def rewrite_query(state):

    prompt = get_rewrite_prompt(
        state["question"]
    )

    rewritten = llm.invoke(prompt)

    return {
        "rewritten_question": rewritten.strip()
    }
