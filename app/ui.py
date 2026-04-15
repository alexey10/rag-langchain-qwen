import sys
import os
import time
import logging
import streamlit as st
from dotenv import load_dotenv

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
logging.basicConfig(
    filename="rag_traces.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)

# -------------------------------
# Fix import path
# -------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# -------------------------------
# Imports
# -------------------------------
from langsmith import traceable
from app.retrieval.retriever import get_retriever
from app.chains.rag_chain import build_rag_chain
from app.ingestion.ingest import run_ingestion

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="RAG Demo", layout="wide")

st.title("🧠 RAG Demo (LangChain + Qwen)")
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

# -------------------------------
# Load components
# -------------------------------
@st.cache_resource
def load_components():
    retriever = get_retriever()
    chain = build_rag_chain(retriever)
    return retriever, chain

retriever, qa_chain = load_components()

# -------------------------------
# 🔍 Retrieval step (traced)
# -------------------------------
@traceable(name="retrieval_step")
def retrieve_docs(query):
    docs = retriever.get_relevant_documents(query)

    # Local logging
    sources = [doc.metadata.get("source", "unknown") for doc in docs]
    logging.info(f"[RETRIEVAL] Query: {query}")
    logging.info(f"[RETRIEVAL] Sources: {sources}")

    return docs

# -------------------------------
# 🧠 Generation step (traced)
# -------------------------------
@traceable(name="generation_step")
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

    answer = qa_chain.invoke(prompt)

    # Local logging
    logging.info(f"[GENERATION] Answer: {str(answer)[:200]}")

    return answer

# -------------------------------
# 🚀 Full RAG pipeline (traced)
# -------------------------------
@traceable(name="rag_query")
def run_query(query):
    start_time = time.time()

    docs = retrieve_docs(query)
    answer = generate_answer(query, docs)

    latency = round(time.time() - start_time, 2)

    # Attach metadata to trace
    return {
        "result": answer,
        "source_documents": docs,
        "metadata": {
            "latency": latency,
            "query_length": len(query),
            "num_docs": len(docs),
        },
    }

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
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        start = time.time()

        result = run_query(user_input)

        end = time.time()

        answer = result.get("result", "")
        sources = result.get("source_documents", [])

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
# Latency
# -------------------------------
if st.session_state.latency:
    st.caption(f"⏱️ Response time: {st.session_state.latency}s")

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
