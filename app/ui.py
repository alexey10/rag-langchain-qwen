import sys
import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import logging
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from app.graph.rag_graph import rag_graph
from app.ingestion.ingest import run_ingestion
from app.config import DATA_PATH
from app.utils.document_utils import (
    get_indexed_documents
)

from app.evaluation.eval_dashboard import (
    load_latest_evaluation
)

from app.evaluation.eval_dashboard import (
    get_recent_runs
)

from app.evaluation.eval_dashboard import (
    get_evaluation_history
)

from app.evaluation.run_eval import (
    run_evaluation
)

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
# Page config
# -------------------------------
st.set_page_config(page_title="Agentic Knowledge Assistant", layout="wide")

st.title("🧠 Agentic Knowledge Assistant (LangChain + Qwen)")
st.markdown("Ask questions about your documents")

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

if "rewritten_question" not in st.session_state:
    st.session_state.rewritten_question = ""

if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

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
        st.session_state.validation = ""
        st.session_state.rewritten_question = ""
        st.session_state.retry_count = 0
        st.success("Chat cleared")      

    if st.sidebar.checkbox(
        "Enable Workspace Reset"
    ):
    
        if st.sidebar.button(
            "🗑 Clear Workspace"
        ):
    
            clear_workspace()
    
            st.rerun()

# -------------------------------
# Indexed Documents
# -------------------------------
documents = get_indexed_documents()

st.sidebar.subheader(
    f"📄 Indexed Documents ({len(documents)})"
)

if documents:

    for doc in documents:

        file_size = os.path.getsize(
            os.path.join(DATA_PATH, doc)
        )

        st.sidebar.write(
            f"• {doc} ({round(file_size/1024,1)} KB)"
        )

else:

    st.sidebar.caption(
        "No PDF documents found."
    )

# -------------------------------
# Running Evaluation
# -------------------------------
st.sidebar.subheader(
    "🧪 Evaluation"
)

if st.sidebar.button(
    "Run Evaluation"
):
    with st.spinner(
        "Running evaluation..."
    ):
        report = run_evaluation()

    st.success(
        f"Accuracy: "
        f"{report['accuracy']:.1f}%"
    )

report = load_latest_evaluation()


# -------------------------------
# Evaluation Dashboard
# -------------------------------
st.sidebar.subheader(
    "🧪 Last Evaluation"
)

report = load_latest_evaluation()

if report:

    timestamp = datetime.fromisoformat(
        report["timestamp"]
    ) 

    st.sidebar.metric(
        "Accuracy",
        f"{report['accuracy']:.1f}%"
    )

    st.sidebar.write(
        f"Passed: {report['passed']}"
    )

    st.sidebar.write(
        f"Failed: {report['failed']}"
    )

    st.sidebar.write(
        f"Latency: "
        f"{report['average_latency']:.1f}s"
    )

    st.sidebar.caption(
        timestamp.strftime("%Y-%m-%d %H:%M")
    )

else:

    st.sidebar.caption(
        "No evaluation results found."
    )

st.sidebar.subheader(
    "📈 Recent Runs"
)

for run in get_recent_runs():

    st.sidebar.caption(
        f"{run['time']}  •  "
        f"A:{run['accuracy']:.0f}% "
        f"R:{run['retrieval_accuracy']:.0f}%"
    )

# -------------------------------
# Evaluation History
# -------------------------------
history = get_evaluation_history()

if history:

    df = pd.DataFrame(history)

    st.subheader(
        "📊 Quality Trends"
    )

    chart_df = df.set_index("run")[
        [
            "accuracy",
            "retrieval_accuracy"
        ]
    ]

    st.line_chart(chart_df)

    st.subheader(
        "⚡ Performance Trends"    
    )
    
    st.line_chart(
        df.set_index("run")["latency"]
    )

else:

    st.info(
        "No evaluation history available yet."
    )
# -------------------------------
# Document Filter
# -------------------------------

selected_docs = st.sidebar.multiselect(
    "Search Documents",
    options=documents,
    default=documents
)

st.session_state.selected_docs = selected_docs

if selected_docs:

    st.sidebar.caption(
        f"🔎 Searching {len(selected_docs)} document(s)"
    )

    for doc in selected_docs:
        st.sidebar.write(
            f"✓ {doc}"
        )

else:

    st.sidebar.warning(
        "No documents selected."
    )

# -------------------------------
# Upload PDFs
# -------------------------------


uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    saved_count = 0

    for uploaded_file in uploaded_files:

        save_path = os.path.join(
            DATA_PATH,
            uploaded_file.name
        )

        if os.path.exists(save_path):
            st.sidebar.warning(
                f"{uploaded_file.name} already exists."
            )

        else:
            with open(save_path, "wb") as f:
                f.write(
                    uploaded_file.getbuffer()
                )
            saved_count += 1

    if saved_count > 0:
        st.sidebar.success(
            f"{saved_count} file(s) uploaded."
        )

    st.sidebar.info(
        "Click Re-index to update the vector database."
    )


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
                    "question": user_input,
                    "retry_count": 0,
                    "selected_docs": st.session_state.selected_docs,
                }
            )

            answer = result.get("answer", "")
            sources = result.get("documents", [])

            validation = result.get(
                "validation",
                "UNKNOWN"
            )

            rewritten_question = result.get(
                "rewritten_question",
                ""
            )
            retry_count = result.get(
                "retry_count",
                0
            )

        except Exception as e:

            st.error(f"Graph execution failed: {e}")
            logging.exception("Graph execution failed")

            answer = "An error occurred while processing your request."
            sources = []
            validation = "ERROR"
            rewritten_question = ""
            retry_count = 0

        end = time.time()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.sources = sources

        st.session_state.validation = validation

        st.session_state.rewritten_question = rewritten_question

        st.session_state.retry_count = retry_count
        
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

# -------------------------------
# Retry count
# -------------------------------
if st.session_state.get("retry_count", 0):
    st.caption(
        f"🔄 Retry attempts: {st.session_state.retry_count}"
    )

# -------------------------------
# Rewrite
# -------------------------------

if st.session_state.get("rewritten_question"):

    st.info(
        f"🔍 Search Query: "
        f"{st.session_state.rewritten_question}"
    )

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
