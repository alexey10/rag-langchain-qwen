import subprocess
import time
import modal

app = modal.App("deepverified-api")

volume = modal.Volume.from_name("deepverified-models")
data_volume = modal.Volume.from_name(
    "deepverified-api-data",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zstd")
    .pip_install_from_requirements("requirements-api.txt")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh"
    )
    .add_local_dir("app", "/root/app")
)


@app.cls(
    image=image,
    gpu="L4",
    volumes={
        "/workspace/ollama": volume,
        "/workspace/data": data_volume,
    },
    env={
        "OLLAMA_MODELS": "/workspace/ollama",
        "OLLAMA_HOST": "127.0.0.1:11434",
        "CHROMA_PATH": "/workspace/data/chroma",
        "KNOWLEDGE_BASE_DB_PATH": "/workspace/data/knowledge_bases.sqlite3",
        "DOCUMENT_STORAGE_PATH": "/workspace/data/knowledge_base_documents",
    },
    secrets=[modal.Secret.from_name("deepverified-api-key")],
    scaledown_window=300,
    max_containers=1,
    timeout=600,
)
class DeepVerifiedAPI:

    @modal.enter()
    def start_ollama(self):
        print("Starting Ollama...")

        self.ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for _ in range(30):
            result = subprocess.run(
                [
                    "curl",
                    "-s",
                    "http://127.0.0.1:11434/api/tags",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print("Ollama is ready.")
                break

            time.sleep(1)
        else:
            raise RuntimeError("Ollama did not start")

        models = subprocess.check_output(
            ["ollama", "list"],
            text=True,
        )

        print("Available models:")
        print(models)

        if "qwen3:latest" not in models:
            print("qwen3:latest not found. Pulling model...")
            subprocess.run(
                ["ollama", "pull", "qwen3:latest"],
                check=True,
            )
        else:
            print("qwen3:latest already available.")

        # NEW: load model into memory
        print("Warming qwen3:latest...")

        subprocess.run(
            [
                "ollama",
                "run",
                "qwen3:latest",
                "Reply with exactly OK",
            ],
            check=True,
        )

        print("qwen3:latest is warm.")


    @modal.asgi_app()
    def web(self):
        import sys

        sys.path.insert(0, "/root")

        from app.api.main import app as fastapi_app

        return fastapi_app
