from app.api.knowledge_base_registry import (
    initialize_registry,
    register_knowledge_base,
)


def main():
    initialize_registry()

    entries = [
        ("default", "Default knowledge base"),
        ("test-kb", "Knowledge-base isolation test"),
    ]

    for knowledge_base_id, name in entries:
        result = register_knowledge_base(
            knowledge_base_id,
            name,
        )
        print(
            f"{result.id}: {result.name} "
            f"({result.status})"
        )


if __name__ == "__main__":
    main()
