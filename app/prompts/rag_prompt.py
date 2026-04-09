from langchain.prompts import PromptTemplate

def get_prompt():
    template = """
You are a helpful assistant. Use ONLY the context below.

Context:
{context}

Question:
{question}

Rules:
- If unknown, say "I don't know"
- Be precise

Answer:
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
