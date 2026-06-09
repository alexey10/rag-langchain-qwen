import time
import json
from app.graph.rag_graph import rag_graph

with open(
    "data/evaluation/eval_set.json"
) as f:
    dataset = json.load(f)

results = []

for item in dataset:

    start = time.time()

    result = rag_graph.invoke(
        {
            "question": item["question"],
            "retry_count": 0,
            "selected_docs": []
        }
    )

    elapsed = time.time() - start

    print(
        f"{item['question']} "
        f"({elapsed:.1f}s)"
    )

    answer = result["answer"]

    passed = (
        item["expected_answer"]
        in answer
    )

    results.append(
        {
            "question": item["question"],
            "pass": passed,
            "latency": elapsed
        }
    )

    print(
        f"{'PASS' if passed else 'FAIL'} | "
        f"{item['question']}"
    )

    print(f"Expected: {item['expected_answer']}")
    print(f"Answer: {answer}")
    print()


total = len(results)

avg_latency = (
    sum(r["latency"] for r in results)
    / total
)

passed = sum(
    1
    for r in results
    if r["pass"]
)

accuracy = (
    passed / total
) * 100

print("\n====================")
print("Evaluation Results")
print("====================")
print(f"Questions: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Accuracy: {accuracy:.1f}%")
print(f"Average Latency: {avg_latency:.2f}s")
print("====================")
