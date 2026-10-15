from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class APIClient:
    client_id: str
    allowed_knowledge_bases: frozenset[str]


# Knowledge bases currently available in the local Chroma collection.
KNOWN_KNOWLEDGE_BASES = frozenset({
    "default",
    "test-kb",
})


# Client policy configuration.
# API key values are read from environment variables, not stored here.
CLIENT_DEFINITIONS = {
    "dev_local": {
        "api_key_env": "DEEPVERIFIED_API_KEY",
        "allowed_knowledge_bases": frozenset({
            "default",
            "test-kb",
        }),
    },
    "dev_default_only": {
        "api_key_env": "DEEPVERIFIED_DEFAULT_ONLY_API_KEY",
        "allowed_knowledge_bases": frozenset({
            "default",
        }),
    },
}


class ClientRegistryConfigurationError(RuntimeError):
    pass


class KnowledgeBaseAccessError(Exception):
    pass


def get_configured_clients_by_api_key():
    clients = {}

    for client_id, config in CLIENT_DEFINITIONS.items():
        api_key = os.getenv(config["api_key_env"])

        # A client is inactive until its API key is configured.
        if not api_key:
            continue

        if api_key in clients:
            raise ClientRegistryConfigurationError(
                "Multiple clients use the same API key"
            )

        clients[api_key] = APIClient(
            client_id=client_id,
            allowed_knowledge_bases=config[
                "allowed_knowledge_bases"
            ],
        )

    return clients


def get_client_by_api_key(api_key):
    clients = get_configured_clients_by_api_key()

    if not clients:
        raise ClientRegistryConfigurationError(
            "No API clients are configured"
        )

    return clients.get(api_key)


def authorize_knowledge_base(client, knowledge_base_id):
    # Treat unknown and unauthorized KBs identically.
    if (
        knowledge_base_id not in KNOWN_KNOWLEDGE_BASES
        or knowledge_base_id not in client.allowed_knowledge_bases
    ):
        raise KnowledgeBaseAccessError()
