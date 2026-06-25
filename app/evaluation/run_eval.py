import time
import json
import os
from datetime import datetime
from app.graph.rag_graph import rag_graph


def invoke_with_timeout(item, timeout=60):
    result_container = {}
    error_container = {}

    def target():
        try:
            result_container["result"] = rag_graph.invoke({
                "question": item["question"],
                "retry_count": 0,
                "selected_docs": []
            })
        except Exception as e:
            error_container["error"] = e

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"TIMEOUT | {item['question']}")
        return None

    if "error" in error_container:
        raise error_container["error"]

    return result_container["result"]


def evaluate_item(item):
    print(f"START | {item['question']}")
    start = time.time()

    print("INVOKING graph...")
    try:
        result = rag_graph.invoke({
            "question": item["question"],
            "retry_count": 0,
            "selected_docs": []
        })
    except Exception as e:
        print(f"ERROR | {item['question']} — {e}")
        return {
            "question": item["question"],
            "pass": False,
            "latency": time.time() - start,
            "answer": f"ERROR: {e}",
            "expected": item["expected_answer"]
        }

    print(result["documents"][0].metadata)

    elapsed = time.time() - start

    print(f"GRAPH RETURNED | {elapsed:.1f}s")
    answer = result["answer"]
    retrieved_sources = []

    for doc in result["documents"]:
    
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
    print(result.keys())

    return {
        "question": item["question"],
        "pass": passed,
        "latency": elapsed,
        "answer": answer,
        "retrieval_pass": retrieval_pass,
        "expected": item["expected_answer"]
    }

def run_evaluation():
    # --- Load dataset (once) ---
    with open("data/evaluation/eval_set.json") as f:
        dataset = json.load(f)

    # --- Warm up ---
    print("Warming up graph...")
    rag_graph.invoke({
        "question": dataset[0]["question"],
        "retry_count": 0,
        "selected_docs": []
    })
    print("Warm-up complete.\n")

    # --- Run evaluation ---
    results = []
    for item in dataset:
        result = evaluate_item(item)
        results.append(result)

    # --- Build report ---
    total = len(results)
    avg_latency = sum(r["latency"] for r in results) / total
    passed_count = sum(1 for r in results if r["pass"])
    accuracy = (passed_count / total) * 100
    retrieval_passed = sum(1 for r in results if r["retrieval_pass"])
    retrieval_accuracy = (retrieval_passed / total) * 100

    report = {
        "timestamp": datetime.now().isoformat(),
        "questions": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "accuracy": accuracy,
        "average_latency": avg_latency,
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
    print(f"Answer Accuracy: " f"{accuracy:.1f}%")
    print(f"Retrieval Accuracy: " f"{retrieval_accuracy:.1f}%")
    print("====================")

    return report


if __name__ == "__main__":
    run_evaluation()
