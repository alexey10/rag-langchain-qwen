def get_validation_prompt(
    question,
    answer
):

    return f"""
Evaluate whether the answer
adequately addresses the question.

Question:
{question}

Answer:
{answer}

Return only:

PASS

or

RETRY
"""
