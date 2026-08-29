from fastapi import Header, HTTPException, Request
from app.config import DEEPVERIFIED_API_KEY


def verify_api_key(
    request: Request,
    authorization: str | None = Header(default=None),
):
    if not DEEPVERIFIED_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server",
        )
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

    if api_key != DEEPVERIFIED_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    # Set client identity after successful validation
    # Replace "dev_local" with a key→client registry when multi-tenant
    request.state.client_id = "dev_local"
