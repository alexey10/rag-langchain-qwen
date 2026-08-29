import time
import json
import logging
import os
from uuid import uuid4
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

LOG_DIR = "/tmp/deepverified"
os.makedirs(LOG_DIR, exist_ok=True)

file_handler = logging.FileHandler(f"{LOG_DIR}/api.log")
file_handler.setFormatter(logging.Formatter("%(message)s"))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("deepverified")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid4())
        start = time.perf_counter()
        request.state.request_id = request_id

        logger.info(json.dumps({
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        }))

        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            logger.info(json.dumps({
                "event": "request_completed",
                "request_id": request_id,
                "client_id": getattr(request.state, "client_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            }))

            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            logger.error(json.dumps({
                "event": "request_failed",
                "request_id": request_id,
                "client_id": getattr(request.state, "client_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "latency_ms": latency_ms,
                "error": str(exc),
            }))
            raise
