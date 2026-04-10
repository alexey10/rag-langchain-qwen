# 🚀 Production-Ready RAG System with LangChain, Qwen & Chroma

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

```
Documents → Chunking → Embeddings → Vector Store (Chroma)
                                      ↓
User Query → Retriever → (Optional Reranker) → Prompt → LLM (Qwen3)
                                                              ↓
                                                           Answer
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
rag-langchain-qwen/
├── app/
│   ├── ingestion/        # Document loading & chunking
│   ├── embeddings/       # Embedding models (BGE)
│   ├── vectorstore/      # Chroma integration
│   ├── retrieval/        # Retriever + reranker
│   ├── llm/              # Qwen (Ollama wrapper)
│   ├── chains/           # RAG pipeline
│   ├── prompts/          # Prompt templates
│   └── main.py           # CLI entry point
│
├── data/docs/            # Source documents
├── chroma_db/            # Persistent vector DB
├── requirements.txt
└── README.md
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

## 📈 Future Improvements

* 🔹 Hybrid search (vector + keyword)
* 🔹 LangGraph for multi-step / agentic RAG
* 🔹 Evaluation (RAGAS / TruLens)
* 🔹 FastAPI service with streaming responses
* 🔹 Multi-document reasoning workflows

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
