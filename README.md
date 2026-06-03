# 🚀 Agentic RAG System with LangGraph, LangSmith, ChromaDB, and Qwen

A modular, local-first Retrieval-Augmented Generation (RAG) system designed for high-quality question answering over private documents. Built with a focus on **scalability, observability, and real-world deployment patterns**.

---

## 🧠 Overview

This project demonstrates how to build a production-style **Agentic RAG** system using open-source components.

### Technology Stack

- **LLM:** Qwen3 (via Ollama)
- **Embeddings:** BGE (`bge-large-en`)
- **Vector Store:** ChromaDB
- **Agent Framework:** LangGraph
- **Observability:** LangSmith
- **UI:** Streamlit

The system ingests documents, indexes them into a vector database, rewrites user queries for improved retrieval, generates grounded responses, and validates answers before returning them to the user.

---

## 🏗️ Architecture

```text
Documents
   │
   ▼
Chunking
   │
   ▼
Embeddings
   │
   ▼
ChromaDB

Question
   │
   ▼
Rewrite Query
   │
   ▼
Retrieve
   │
   ▼
Generate
   │
   ▼
Validate
   │
   ▼
Answer
```

---

## 👤 User Flow

1. User submits a question
2. Query Rewrite node optimizes the search query
3. Retriever searches ChromaDB for relevant chunks
4. Qwen generates an answer using retrieved context
5. Validation node evaluates answer quality
6. Final answer is returned to the user

### Observability

- LangSmith tracing
- Local logging
- Retrieved context inspection

---

## ⚙️ Key Features

### ✅ Agentic RAG Workflow

- Query rewriting for improved retrieval quality
- Semantic retrieval using ChromaDB
- Grounded answer generation using Qwen
- LLM-based answer validation

### ✅ Modular Design

- Clean separation of concerns
- Easy component replacement
- Extensible architecture

### ✅ Local-First & Privacy-Friendly

- Runs locally via Ollama
- No external LLM API required
- Suitable for private document collections

### ✅ Observability

- LangSmith tracing
- Local logging
- Retrieval inspection
- Debug-friendly workflow visualization

### ✅ Production-Oriented Patterns

- Persistent ChromaDB storage
- Config-driven architecture
- Streamlit UI
- Agent workflow orchestration with LangGraph

---

## 📂 Project Structure

```text
rag-langchain-qwen/
│
├── app/
│   ├── ui.py
│   ├── config.py
│   │
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes.py
│   │   └── rag_graph.py
│   │
│   ├── ingestion/
│   │   ├── ingest.py
│   │   ├── loader.py
│   │   └── splitter.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py
│   │
│   ├── embeddings/
│   │   └── embedding.py
│   │
│   ├── llm/
│   │   └── qwen_llm.py
│   │
│   ├── prompts/
│   │   └── rag_prompt.py
│   │
│   └── chains/
│       └── rag_chain.py  # Legacy implementation
│
├── data/
├── chroma_db/
├── logs/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 LangGraph Workflow

```text
START
  │
  ▼
rewrite_query
  │
  ▼
retrieve
  │
  ▼
generate
  │
  ▼
validate
  │
  ▼
END
```

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/alexey10/rag-langchain-qwen.git
cd rag-langchain-qwen
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file:

```env
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=agentic-rag

HF_TOKEN=your_huggingface_token
```

### 4. Start Ollama

```bash
ollama serve
ollama run qwen3
```

### 5. Add Documents

Place files in:

```text
data/docs/
```

### 6. Build the Vector Index

```bash
python -m app.main
```

### 7. Launch the Application

```bash
streamlit run app/ui.py
```

Open:

```text
http://localhost:8501
```

---

## 🔍 Example Output

```text
Question:
What inflation changes are expected in 2027?

Answer:
Global headline inflation is expected to decline to 3.4% in 2027,
down from 3.8% in 2026.

Search Query:
What is the IMF forecast for global inflation in 2027?

Validation:
PASS
```

---

## 📊 Current Status

| Item | Status |
|--------|--------|
| Basic RAG (Qwen + Chroma) | ✅ Complete |
| Streamlit UI | ✅ Complete |
| LangSmith Observability | ✅ Complete |
| Query Rewrite Node | ✅ Complete |
| LLM Validation Node | ✅ Complete |
| LangGraph Agentic RAG | ✅ Complete |
| Local Deployment | ✅ Complete |
| GitHub Portfolio Project | ✅ Complete |

### Agent Maturity

| Level | Capability |
|---------|---------|
| Traditional RAG | Retrieve → Generate |
| Agentic RAG v1 | Rewrite → Retrieve → Generate → Validate |
| Agentic RAG v2 | Retry on Failed Validation |
| Agentic RAG v3 | Multiple Retrieval Strategies |
| Agentic RAG v4 | Tool Calling + Planning |

---

## 📈 Future Improvements

### Next Iteration

- 🔹 Self-correcting validation loop
- 🔹 Hybrid Search (Vector + BM25)
- 🔹 Hallucination Detection
- 🔹 Multi-document Reasoning
- 🔹 Citation-aware Responses
- 🔹 RAG Evaluation with RAGAS
- 🔹 FastAPI Inference Service
- 🔹 Hugging Face Deployment

---

## 💡 Key Learnings

- RAG performance depends more on retrieval quality than model size
- Query rewriting can significantly improve retrieval quality
- Chunking strategy impacts answer accuracy
- Observability is critical for debugging production AI systems
- Validation layers improve trust and answer quality

---

## 🎯 Use Cases

- API documentation assistant
- Internal knowledge base search
- Compliance and policy Q&A
- Partner integration support
- Financial and research document analysis

---

## 👤 Author

**Alexey Piskovatskov**

Focus areas:
- AI Systems
- Program Management
- Agentic Workflows
- Scalable Architectures

---

## 📄 License

MIT License
