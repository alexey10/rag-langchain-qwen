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

graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("validate", validate)

graph.set_entry_point("rewrite_query")

graph.add_edge(
    "rewrite_query",
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    "validate"
)

graph.add_edge(
    "validate",
    END
)

rag_graph = graph.compile()
