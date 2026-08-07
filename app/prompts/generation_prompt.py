def get_generation_prompt(
    context,
    question
):

    return f"""
You are a financial research assistant.

Answer the question using only the
provided context.

Requirements:
- Be concise.
- Lead with the answer.
- Do not speculate.
- Use one or two sentences.

Context:
{context}

Question:
{question}
"""
