# app/observability/langsmith_config.py

import os

from dotenv import load_dotenv


def configure_langsmith():
    load_dotenv()

    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing LANGCHAIN_API_KEY in environment"
        )

    # Keep the existing project compatible with LangSmith 0.8.x.
    os.environ["LANGCHAIN_API_KEY"] = api_key

    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv(
        "LANGCHAIN_TRACING_V2",
        "true",
    )

    os.environ["LANGCHAIN_PROJECT"] = os.getenv(
        "LANGCHAIN_PROJECT",
        "rag-demo",
    )

    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT",
        "https://api.smith.langchain.com",
    )
