import json
import os


def load_latest_evaluation():

    path = (
        "data/evaluation/results/"
        "latest_eval.json"
    )

    if not os.path.exists(path):
        return None

    with open(path) as f:
        return json.load(f)

def get_recent_runs():

    history_path = (
        "data/evaluation/results/history"
    )

    if not os.path.exists(history_path):
        return []

    files = sorted(
        os.listdir(history_path),
        reverse=True
    )

    return files[:2]
