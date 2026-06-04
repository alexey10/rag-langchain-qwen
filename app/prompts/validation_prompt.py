def get_validation_prompt(
    question,
    answer
):

    return f"""
You are validating an answer produced by a RAG system.

Question:
{question}

Answer:
{answer}

Return only:

PASS

or

RETRY

PASS if:
- The answer attempts to answer the question.
- The answer is clear and relevant.
- The answer does not contain obvious contradictions.

RETRY if:
- The answer is empty.
- The answer does not address the question.
- The answer appears unrelated to the question.

Return only PASS or RETRY.
"""
