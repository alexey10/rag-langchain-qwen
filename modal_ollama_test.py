import modal
import subprocess
import time

app = modal.App("deepverified-ollama-test")

volume = modal.Volume.from_name("deepverified-models")

image = (
    modal.Image.debian_slim()
    .apt_install("curl", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh"
    )
)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/workspace/ollama": volume},
    env={
        "OLLAMA_MODELS": "/workspace/ollama",
        "OLLAMA_HOST": "127.0.0.1:11434",
    },
)
def test_ollama():
    print("Starting Ollama...")

    process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for Ollama API
    for _ in range(30):
        result = subprocess.run(
            ["curl", "-s", "http://127.0.0.1:11434/api/tags"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Ollama is ready.")
            break

        time.sleep(1)
    else:
        raise RuntimeError("Ollama did not start")

    print("Existing models:")
    subprocess.run(["ollama", "list"])

    # Pull only if Qwen isn't already in the persistent volume
    models = subprocess.check_output(
        ["ollama", "list"],
        text=True,
    )

    if "qwen3" not in models:
        print("Pulling qwen3:latest...")
        subprocess.run(
            ["ollama", "pull", "qwen3:latest"],
            check=True,
        )

    print("Running inference...")

    result = subprocess.run(
        [
            "ollama",
            "run",
            "qwen3:latest",
            "Reply with exactly: OK",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    print("MODEL RESPONSE:")
    print(result.stdout)

    return result.stdout
