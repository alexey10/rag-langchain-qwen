from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from app.api.auth import verify_api_key


def get_api_key(request: Request) -> str:
    """Use API key as rate limit identifier instead of IP."""
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Bearer "):
        return authorization.removeprefix("Bearer ").strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_api_key)
