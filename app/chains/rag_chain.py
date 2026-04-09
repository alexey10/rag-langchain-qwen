from langchain.chains import RetrievalQA
from app.llm.qwen_llm import get_llm
from app.prompts.rag_prompt import get_prompt

def build_rag_chain(retriever):
    llm = get_llm()
    prompt = get_prompt()

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
