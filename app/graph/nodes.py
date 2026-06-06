import logging

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

    selected_docs = state.get(
        "selected_docs",
        []
    )

    if selected_docs:

        docs = [
            doc
            for doc in docs
            if any(
                selected_doc
                in doc.metadata.get(
                    "source",
                    ""
                )
                for selected_doc
                in selected_docs
            )
        ]

    print(f"Retrieved docs: {len(docs)}")

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

# Validation Node

def validate(state):

    prompt = get_validation_prompt(
        state["question"],
        state["answer"]
    )

    result = llm.invoke(prompt)

    retry_count = state.get(
        "retry_count",
        0
    )

    print(
        f"VALIDATION ATTEMPT {retry_count + 1}: {result}"
    )

    logging.info(
        f"VALIDATION ATTEMPT {retry_count + 1}: {result}"
    )

    validation = result.strip().upper()

    if "PASS" in validation:
        return {
            "validation": "PASS"
        }

    return {
        "validation": "RETRY",
        "retry_count": retry_count + 1,
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
