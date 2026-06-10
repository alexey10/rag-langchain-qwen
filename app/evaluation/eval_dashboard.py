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

def get_evaluation_history():

    history_path = (
        "data/evaluation/results/history"
    )

    if not os.path.exists(history_path):
        return []

    files = sorted(
        os.listdir(history_path)
    )

    history = []

    for file in files:

        with open(
            os.path.join(
                history_path,
                file
            )
        ) as f:

            report = json.load(f)
    
        timestamp = file.replace(
            "eval_",
            ""
        ).replace(
            ".json",
            ""
        )
    
        label = (
            timestamp[5:7]      # month
            + "/"
            + timestamp[8:10]   # day
            + " "
            + timestamp[11:13]  # hour
            + ":"
            + timestamp[13:15]  # minute
        )
    
        history.append(
            {
                "run": label,
                "accuracy": report["accuracy"],
                "latency": report[
                    "average_latency"
                ]
            }
        )

    return history
