import modal

app = modal.App("deepverified-gpu-test")


@app.function(gpu="L4")
def test_gpu():
    import subprocess

    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)
