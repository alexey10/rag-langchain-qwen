from langgraph.graph import StateGraph
from langgraph.graph import END

from app.graph.state import RAGState
from app.graph.nodes import (
    rewrite_query,
    retrieve,
    generate,
    validate,
)

graph = StateGraph(RAGState)

def start(state):

    return {}


def route_after_start(state):

    if state.get("enable_rewrite", True):
        return "rewrite"

    return "retrieve"


def route_validation(state):

    if state["validation"] == "PASS":
        return "pass"

    if state.get("retry_count", 0) >= 2:
        return "pass"

    return "retry"


def route_after_generate(state):

    if state.get("enable_validation", True):
        return "validate"

    return "pass"


graph.add_node("start", start)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("validate", validate)

graph.set_entry_point("start")

graph.add_conditional_edges(
    "start",
    route_after_start,
    {
        "rewrite": "rewrite_query",
        "retrieve": "retrieve",
    }
)

graph.add_edge(
    "rewrite_query",
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_conditional_edges(
    "generate",
    route_after_generate,
    {
        "validate": "validate",
        "pass": END,
    }
)

graph.add_conditional_edges(
    "validate",
    route_validation,
    {
        "pass": END,
        "retry": "rewrite_query",
    }
)

rag_graph = graph.compile()
