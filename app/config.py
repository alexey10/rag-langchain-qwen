import os

from dotenv import load_dotenv
load_dotenv()

DATA_PATH = "data/docs"
CHROMA_PATH = "chroma_db"

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "127.0.0.1:11434",
)

EMBEDDING_MODEL = "BAAI/bge-large-en"
#LLM_MODEL = "deepseek-r1:8b"
#LLM_MODEL = "qwen3.5:9b"
LLM_MODEL = "qwen3"
LLM_NUM_PREDICT = int(
    os.getenv("LLM_NUM_PREDICT", "128")
)
LLM_KEEP_ALIVE = os.getenv(
    "LLM_KEEP_ALIVE",
    "30m"
)
LLM_REASONING = (
    os.getenv("LLM_REASONING", "false").lower()
    == "true"
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4

REWRITE_CACHE_ENABLED = (
    os.getenv("REWRITE_CACHE_ENABLED", "true").lower()
    == "true"
)

DEEPVERIFIED_API_KEY = os.getenv("DEEPVERIFIED_API_KEY")

KNOWLEDGE_BASE_DB_PATH = os.getenv(
    "KNOWLEDGE_BASE_DB_PATH",
    "data/knowledge_bases.sqlite3",
)

DOCUMENT_STORAGE_PATH = os.getenv(
    "DOCUMENT_STORAGE_PATH",
    "data/knowledge_base_documents",
)

MAX_DOCUMENT_UPLOAD_BYTES = int(
    os.getenv("MAX_DOCUMENT_UPLOAD_BYTES", str(20 * 1024 * 1024))
)
