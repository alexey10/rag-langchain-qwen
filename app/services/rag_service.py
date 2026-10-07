from app.graph.rag_graph import rag_graph
from app.gateway.availability import get_available_model_config


class RAGService:

    def answer(
        self,
        messages,
        knowledge_base_id,
        model,
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

        model_config = get_available_model_config(model)

        if model_config["provider"] != "ollama":
            raise ValueError(
                f"Unsupported provider: "
                f"{model_config['provider']}"
            )

        runtime_model = model_config["model"]

        question = user_messages[-1].content

        result = rag_graph.invoke({
            "question": question,
            "knowledge_base_id": knowledge_base_id,
            "model": runtime_model,
            "enable_rewrite": True,
            "enable_validation": True,
        })

        return result["answer"]
