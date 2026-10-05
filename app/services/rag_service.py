from app.graph.rag_graph import rag_graph


class RAGService:

    def answer(
        self,
        messages,
        knowledge_base_id,
    ):
        if knowledge_base_id != "default":
            raise ValueError(
                f"Unsupported knowledge base: {knowledge_base_id}"
            )

        user_messages = [
            message
            for message in messages
            if message.role == "user"
        ]

        if not user_messages:
            raise ValueError(
                "At least one user message is required"
            )

        question = user_messages[-1].content

        result = rag_graph.invoke({
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "enable_rewrite": True,
            "enable_validation": True,
        })

        return result["answer"]
