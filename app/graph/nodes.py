import logging

from langsmith import traceable

from app.retrieval.retriever import get_retriever
from app.llm.qwen_llm import get_llm

llm = get_llm()

def get_state_llm(state):
    model = state.get("model")

    if model:
        return get_llm(model)

    return llm

from app.prompts.rewrite_prompt import (
    get_rewrite_prompt
)

from app.prompts.generation_prompt import (
    get_generation_prompt
)

from app.prompts.validation_prompt import (
    get_validation_prompt
)

from app.observability.langfuse_client import langfuse

from app.observability.tracing import (
    traced_node
)

from app.utils.rewrite_cache import (
    get_cached_rewrite,
    save_cached_rewrite,
)

#Retrieval Node

@traced_node
def retrieve(state):

    query = state.get(
        "rewritten_question",
        state["question"],
    )

    knowledge_base_id = state.get(
        "knowledge_base_id",
        "default",
    )

    retriever = get_retriever(
        knowledge_base_id
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

@traced_node
def generate(state):

    context = "\n\n".join(
        doc.page_content
        for doc in state["documents"]
    )

    prompt = get_generation_prompt(
        context,
        state["question"]
    )

    state_llm = get_state_llm(state)

    answer = state_llm.invoke(prompt)

    return {
        "answer": answer
    }

#Validation Node

# Validation Node

@traced_node
def validate(state):

    prompt = get_validation_prompt(
        state["question"],
        state["answer"]
    )

    state_llm = get_state_llm(state)

    result = state_llm.invoke(prompt)

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

@traced_node
def rewrite_query(state):

    question = state["question"]
    cached_rewrite = get_cached_rewrite(question)

    if cached_rewrite:
        return {
            "rewritten_question": cached_rewrite,
            "rewrite_cache_hit": True
        }

    prompt = get_rewrite_prompt(
        question
    )

    state_llm = get_state_llm(state)

    rewritten = state_llm.invoke(prompt)
    rewritten = rewritten.strip()
    save_cached_rewrite(
        question,
        rewritten
    )

    return {
        "rewritten_question": rewritten,
        "rewrite_cache_hit": False
    }
