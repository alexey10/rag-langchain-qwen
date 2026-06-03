def get_rewrite_prompt(question):

    return f"""
You are an expert search query optimizer.

Rewrite the user's question to maximize
retrieval quality.

Rules:
- Preserve intent.
- Replace vague wording with domain-specific terms.
- Return only the rewritten query.

Question:
{question}
"""
