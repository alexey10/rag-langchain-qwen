# 🚀 Agentic RAG system built with LangGraph, LangSmith, ChromaDB, and Qwen featuring query rewriting, retrieval, answer generation, and LLM-based validation.

A modular, local-first Retrieval-Augmented Generation (RAG) system designed for high-quality question answering over private documents. Built with a focus on **scalability, observability, and real-world deployment patterns**.

---

## 🧠 Overview

This project demonstrates how to build a **production-style RAG pipeline** using open-source components:

* **LLM**: Qwen3 (via Ollama)
* **Embeddings**: BGE (`bge-large-en`)
* **Vector Store**: Chroma
* **Orchestration**: LangChain
* **Observability**: LangSmith

The system ingests documents, indexes them into a vector database, retrieves relevant context at query time, and generates grounded responses using an LLM.

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
## User Flow

1. User submits a question
2. Query Rewrite node optimizes the search query
3. Retriever searches ChromaDB for relevant chunks
4. Qwen generates an answer using retrieved context
5. Validation node evaluates answer quality
6. Final answer is returned to the user

Observability:
- LangSmith tracing
- Local logging
- Retrieved context inspection
```
---

## ⚙️ Key Features

### ✅ Modular Design

* Clean separation of concerns:

  * ingestion
  * embeddings
  * retrieval
  * generation
* Easy to extend or swap components (LLM, vector DB, etc.)

---

### ✅ Retrieval-First Architecture

* Focus on **retrieval quality over model size**
* Configurable chunking strategy (size + overlap)
* Supports metadata filtering and extensibility

---

### ✅ Local-First & Privacy-Friendly

* Runs fully locally using Ollama
* No external API dependencies required
* Suitable for sensitive data (e.g., internal docs, APIs)

---

### ✅ Production-Oriented Patterns

* Persistent vector store (Chroma)
* Config-driven architecture
* Ready for API layer (FastAPI) and scaling

---

### ✅ (Optional) Reranking Layer

* Supports cross-encoder reranking (BGE reranker)
* Improves retrieval precision for complex queries

---

## 📂 Project Structure

```

```text
rag-langchain-qwen/
│
├── app/
│   │
│   ├── ui.py                    # Streamlit application
│   ├── config.py                # Configuration settings
│   │
│   ├── graph/
│   │   ├── state.py             # LangGraph state definition
│   │   ├── nodes.py             # Rewrite, Retrieve, Generate, Validate nodes
│   │   └── rag_graph.py         # LangGraph workflow
│   │
│   ├── ingestion/
│   │   ├── ingest.py            # End-to-end ingestion pipeline
│   │   ├── loader.py            # Document loading
│   │   └── splitter.py          # Chunking logic
│   │
│   ├── retrieval/
│   │   └── retriever.py         # ChromaDB retrieval layer
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py      # ChromaDB persistence and loading
│   │
│   ├── embeddings/
│   │   └── embedding.py         # HuggingFace embedding model
│   │
│   ├── llm/
│   │   └── qwen_llm.py          # Ollama / Qwen integration
│   │
│   ├── prompts/
│   │   └── rag_prompt.py        # Prompt templates
│   │
│   └── chains/
│       └── rag_chain.py         # Legacy LangChain RAG chain
│
├── data/                        # Source documents
├── chroma_db/                   # Local vector database
├── logs/                        # Local trace logs
├── .env                         # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

### LangGraph Workflow

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

```

---

## 🚀 Getting Started

### 1. Clone repo

```bash
git clone https://github.com/alexey10/rag-langchain-qwen.git
cd rag-langchain-qwen
```
## 🔐 Environment Variables

Create a `.env` file in the root directory:
LANGCHAIN_API_KEY=your_key
HF_TOKEN=your_token
OPENAI_API_KEY=your_key

---

### 2. Setup environment (Python 3.11 recommended)

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Start LLM (Ollama)

```bash
ollama serve
ollama run qwen3
```

---

### 4. Add documents

Place your files in:

```
data/docs/
```

---

### 5. Run ingestion (one-time)

```bash
python -m app.main
```

(Ensure `run_ingestion()` is enabled for first run)

---

### 6. Query the system

```bash
python -m app.main
```

Example:

```
Ask a question: What are the key risks discussed?
```

---

## 🔍 Example Output

```
Answer:
The document highlights three primary risks: ...

Sources:
- doc_chunk_1
- doc_chunk_2
```

---

## 🧪 Observability (Debugging Retrieval)

To inspect retrieval quality, enable logging in `main.py`:

```python
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:300]}")
```

This helps diagnose:

* irrelevant retrieval
* poor chunking
* missing context

---

## Current Status
| Item                                  | Status          |
| ------------------------------------- | --------------- |
| Basic RAG (Qwen + Chroma + LangChain) | ✅ Complete      |
| Streamlit UI                          | ✅ Complete      |
| LangSmith Observability               | ✅ Complete      |
| LangGraph for Agentic RAG             | ✅ Complete (v1) |
| Local Deployment                      | ✅ Complete      |
| GitHub Portfolio Project              | ✅ Complete      |

| Level           | Capability                               |
| --------------- | ---------------------------------------- |
| Traditional RAG | Retrieve → Generate                      |
| Agentic RAG v1  | Rewrite → Retrieve → Generate → Validate |
| Agentic RAG v2  | Retry on failed validation               |
| Agentic RAG v3  | Multiple retrieval strategies            |
| Agentic RAG v4  | Tool calling + planning                  |

## 📈 Future Improvements

## Next Iteration

* 🔹 Hybrid Search (Vector + BM25)
* 🔹 Query Rewriting and Retrieval Retry
* 🔹 LLM-based Answer Validation
* 🔹 Hallucination Detection
* 🔹 Multi-document Reasoning
* 🔹 RAG Evaluation with RAGAS
* 🔹 FastAPI Inference Service
* 🔹 Hugging Face Deployment

---

## 💡 Key Learnings

* RAG performance depends more on **retrieval quality** than LLM choice
* Chunking strategy significantly impacts answer accuracy
* Reranking provides one of the highest ROI improvements
* Observability is critical for debugging real-world RAG systems

---

## 🎯 Use Cases

* API documentation assistant
* Internal knowledge base search
* Compliance / policy Q&A
* Partner integration support tools

---

## 👤 Author

Built by Alexey Piskovatskov
Focus: AI systems, program management, and scalable architectures

---

## 📄 License

MIT License
