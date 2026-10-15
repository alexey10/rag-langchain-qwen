from fastapi import Header, HTTPException, Request

from app.api.client_registry import (
    APIClient,
    ClientRegistryConfigurationError,
    get_client_by_api_key,
)


def verify_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
) -> APIClient:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing API key",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header",
        )

    api_key = authorization.removeprefix("Bearer ").strip()

    try:
        client = get_client_by_api_key(api_key)
    except ClientRegistryConfigurationError:
        raise HTTPException(
            status_code=500,
            detail="API client registry is not configured",
        )

    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Preserve client identity for structured request logging.
    request.state.client_id = client.client_id

    return client
