import os

DATA_PATH = "data/docs"
CHROMA_PATH = "chroma_db"

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
