import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import logging
import streamlit as st
from dotenv import load_dotenv
from app.graph.rag_graph import rag_graph
from app.ingestion.ingest import run_ingestion

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

# LangSmith config (MUST be before LangChain usage)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-demo")

if not os.getenv("LANGCHAIN_API_KEY"):
    raise ValueError("Missing LANGCHAIN_API_KEY in environment")

# -------------------------------
# Local logging (file-based)
# -------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/rag_traces.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# -------------------------------
# Fix import path
# -------------------------------
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Agentic Knowledge Assistant", layout="wide")

st.title("🧠 Agentic Knowledge Assistant (LangChain + Qwen)")
st.markdown("Ask questions about your documents (Agentic-ready RAG)")

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "latency" not in st.session_state:
    st.session_state.latency = 0

if "validation" not in st.session_state:
    st.session_state.validation = ""


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🔄 Rebuild Index"):
        with st.spinner("Re-indexing documents..."):
            run_ingestion()
        st.success("Index rebuilt")

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.latency = 0
        st.success("Chat cleared")

# -------------------------------
# Chat input
# -------------------------------


user_input = st.chat_input(
    "Ask a question about your documents..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.spinner("Thinking..."):

        start = time.time()

        try:

            result = rag_graph.invoke(
                {
                    "question": user_input
                }
            )

            answer = result.get("answer", "")
            sources = result.get("documents", [])

            validation = result.get(
                "validation",
                "UNKNOWN"
            )

        except Exception as e:

            st.error(f"Graph execution failed: {e}")
            logging.exception("Graph execution failed")

            answer = "An error occurred while processing your request."
            sources = []
            validation = "ERROR"

        end = time.time()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.sources = sources

        st.session_state.validation = validation

        st.session_state.latency = round(
            end - start,
            2
        )
# -------------------------------
# Render chat
# -------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# Latency
# -------------------------------
if st.session_state.latency:
    st.caption(f"⏱️ Response time: {st.session_state.latency}s")


#
# -------------------------------
# Validation Status
# -------------------------------
if st.session_state.validation:

    if st.session_state.validation == "PASS":
        st.success("✅ Validation Passed")

    elif st.session_state.validation == "RETRY":
        st.warning("⚠️ Validation Requested Retry")

# -------------------------------
# Sources display
# -------------------------------
if st.session_state.sources:
    st.subheader("📚 Sources")

    for i, doc in enumerate(st.session_state.sources):
        source_name = doc.metadata.get("source", "unknown")

        st.markdown(f"**Chunk {i+1} — {source_name}**")
        st.write(doc.page_content[:300])

# -------------------------------
# Debug / Observability
# -------------------------------
with st.expander("🔍 Retrieved Context (Debug)"):
    if st.session_state.sources:
        for doc in st.session_state.sources:
            st.write(doc.page_content[:500])
    else:
        st.write("No context retrieved yet.")
