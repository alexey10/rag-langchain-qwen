import modal
import os

app = modal.App("deepverified-volume-test")

volume = modal.Volume.from_name(
    "deepverified-models",
)


@app.function(
    volumes={"/workspace/ollama": volume},
)
def test_volume():
    return os.listdir("/workspace/ollama")
