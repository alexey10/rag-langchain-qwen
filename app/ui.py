import sys
import os
import time
import streamlit as st

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-demo"

from dotenv import load_dotenv

load_dotenv()

os.getenv("LANGCHAIN_API_KEY")

from langsmith import traceable

@traceable(name="rag_query")
def run_query(query, sources=None):
    return qa_chain.invoke(
        query,
        config={
            "metadata": {
                "query_length": len(query),
                "app": "rag-demo",
                "retriever": "chroma",
                "num_sources": len(sources) if sources else 0,
            }
        }
    )

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.retriever import get_retriever
from app.chains.rag_chain import build_rag_chain
from app.ingestion.ingest import run_ingestion

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("🧠 RAG Demo (LangChain + Qwen)")
st.markdown("Ask questions about your documents (RAG-powered)")

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "latency" not in st.session_state:
    st.session_state.latency = 0

# -------------------------------
# Load RAG chain (cached)
# -------------------------------
@st.cache_resource
def load_chain():
    retriever = get_retriever()
    return build_rag_chain(retriever)

qa_chain = load_chain()

# -------------------------------
# Sidebar (controls)
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
        st.success("Chat cleared")

# -------------------------------
# Chat UI
# -------------------------------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        start = time.time()

        result = qa_chain(user_input)

        end = time.time()

        # Handle both chain types
        if isinstance(result, dict):
            answer = result.get("result", "")
            sources = result.get("source_documents", [])
        else:
            answer = result
            sources = []

        # Save to session state
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.session_state.sources = sources
        st.session_state.latency = round(end - start, 2)

# -------------------------------
# Render chat
# -------------------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# Metrics
# -------------------------------
if st.session_state.latency:
    st.caption(f"⏱️ Response time: {st.session_state.latency}s")

# -------------------------------
# Sources (clean UI)
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
