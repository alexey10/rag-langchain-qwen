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

Example:
China is projected to grow at 4.5% in 2026 [Chunk 3].

Context:
{context}

Question:
{question}
"""
