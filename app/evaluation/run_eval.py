import time
import json
import os
import threading
from datetime import datetime
from app.graph.rag_graph import rag_graph

NODE_LATENCY_KEYS = {
    "rewrite_query": "rewrite_latency",
    "retrieve": "retrieve_latency",
    "generate": "generate_latency",
    "validate": "validate_latency",
}


def summarize_node_timings(node_timings):
    summary = {
        key: 0
        for key in NODE_LATENCY_KEYS.values()
    }

    for timing in node_timings:
        node = timing.get("node")
        latency_key = NODE_LATENCY_KEYS.get(node)

        if latency_key:
            summary[latency_key] += timing.get(
                "elapsed_seconds",
                0
            )

    return {
        key: round(value, 3)
        for key, value in summary.items()
    }


EMPTY_NODE_LATENCIES = summarize_node_timings([])


def invoke_graph(item):
    return rag_graph.invoke({
        "question": item["question"],
        "retry_count": 0,
        "selected_docs": [],
        "node_timings": [],
        "enable_validation": False,
        "enable_rewrite": True
    })


def invoke_with_timeout(item, timeout=180):
    if timeout is None:
        return invoke_graph(item)

    result_container = {}
    error_container = {}

    def target():
        try:
            result_container["result"] = invoke_graph(item)
        except Exception as e:
            error_container["error"] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"TIMEOUT | {item['question']}")
        return None

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]


def evaluate_item(item, timeout=None):
    print(f"START | {item['question']}")
    start = time.time()

    print("INVOKING graph...")
    try:
        result = invoke_with_timeout(item, timeout=timeout)
    except Exception as e:
        print(f"ERROR | {item['question']} — {e}")
        return {
            "question": item["question"],
            "pass": False,
            "latency": time.time() - start,
            "answer": f"ERROR: {e}",
            "retrieval_pass": False,
            "retrieved_sources": [],
            "node_timings": [],
            **EMPTY_NODE_LATENCIES,
            "expected": item["expected_answer"]
        }

    if result is None:
        return {
            "question": item["question"],
            "pass": False,
            "latency": time.time() - start,
            "answer": "TIMEOUT",
            "retrieval_pass": False,
            "retrieved_sources": [],
            "node_timings": [],
            **EMPTY_NODE_LATENCIES,
            "expected": item["expected_answer"]
        }

    documents = result.get("documents", [])

    if documents:
        print(documents[0].metadata)

    elapsed = time.time() - start

    print(f"GRAPH RETURNED | {elapsed:.1f}s")
    answer = result.get("answer", "")
    node_timings = result.get("node_timings", [])
    node_latency_summary = summarize_node_timings(
        node_timings
    )
    retrieved_sources = []

    for doc in documents:
    
        source = doc.metadata.get(
            "source",
            ""
        )
    
        filename = source.split("/")[-1]
    
        retrieved_sources.append(
            filename
        )

    retrieval_pass = (
        item["expected_document"]
        in retrieved_sources
    )

    print("ANSWER EXTRACTED")

    passed = item["expected_answer"] in answer
    print(f"{'PASS' if passed else 'FAIL'} | {item['question']} ({elapsed:.1f}s)")
    print(f"Expected: {item['expected_answer']}")
    print(f"Answer:   {answer}\n")
    print(f"Node timings: {node_latency_summary}")
    print(result.keys())

    return {
        "question": item["question"],
        "pass": passed,
        "latency": elapsed,
        "answer": answer,
        "retrieval_pass": retrieval_pass,
        "retrieved_sources": retrieved_sources,
        "node_timings": node_timings,
        **node_latency_summary,
        "expected": item["expected_answer"]
    }

def warm_up_graph(dataset):
    if not dataset:
        return

    print("Warming up graph...")
    rag_graph.invoke({
        "question": dataset[0]["question"],
        "retry_count": 0,
        "selected_docs": [],
        "node_timings": [],
        "enable_validation": False,
        "enable_rewrite": False
    })
    print("Warm-up complete.\n")


def run_evaluation(warm_up=True, timeout=None):
    with open("data/evaluation/eval_set.json") as f:
        dataset = json.load(f)

    if warm_up:
        warm_up_graph(dataset)

    results = []
    for item in dataset:
        result = evaluate_item(item, timeout=timeout)
        results.append(result)

    # existing report code continues...

    # --- Build report ---
    total = len(results)
    avg_latency = sum(r["latency"] for r in results) / total
    passed_count = sum(1 for r in results if r["pass"])
    accuracy = (passed_count / total) * 100
    retrieval_passed = sum(1 for r in results if r["retrieval_pass"])
    retrieval_accuracy = (retrieval_passed / total) * 100
    average_node_latencies = {
        f"average_{latency_key}": (
            sum(r[latency_key] for r in results) / total
        )
        for latency_key in NODE_LATENCY_KEYS.values()
    }

    report = {
        "timestamp": datetime.now().isoformat(),
        "questions": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy": accuracy,
        "average_latency": avg_latency,
        **average_node_latencies,
        "retrieval_accuracy": retrieval_accuracy,
        "results": results
    }

    # --- Save latest_eval.json ---
    os.makedirs("data/evaluation/results", exist_ok=True)
    with open("data/evaluation/results/latest_eval.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Save history ---
    os.makedirs("data/evaluation/results/history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    history_file = f"data/evaluation/results/history/eval_{timestamp}.json"
    with open(history_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n====================")
    print("Evaluation Results")
    print("====================")
    print(f"Questions:        {total}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {total - passed_count}")
    print(f"Accuracy:         {accuracy:.1f}%")
    print(f"Average Latency:  {avg_latency:.2f}s")
    for latency_key, value in average_node_latencies.items():
        print(f"{latency_key}: {value:.2f}s")
    print(f"Answer Accuracy: " f"{accuracy:.1f}%")
    print(f"Retrieval Accuracy: " f"{retrieval_accuracy:.1f}%")
    print("====================")

    return report


if __name__ == "__main__":
    run_evaluation()
