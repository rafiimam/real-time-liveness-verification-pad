"""Simple in-memory sliding-window rate limiter middleware."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Paths that should never be rate-limited harshly
_SKIP_PREFIXES = ("/health", "/app", "/docs", "/openapi.json", "/redoc")


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int | None = None):
        super().__init__(app)
        self.limit = limit if limit is not None else RATE_LIMIT_PER_MINUTE
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def _client_key(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        if request.client:
            return request.client.host or "unknown"
        return "unknown"

    async def dispatch(self, request: Request, call_next) -> Response:
        if not RATE_LIMIT_ENABLED or self.limit <= 0:
            return await call_next(request)

        path = request.url.path or "/"
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)
        # WebSocket upgrade handled separately; HTTP middleware still sees the request
        if path.startswith("/ws/"):
            return await call_next(request)

        key = self._client_key(request)
        now = time.monotonic()
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Rate limit exceeded ({self.limit}/min). Retry later.",
                },
                headers={"Retry-After": "60"},
            )
        window.append(now)
        return await call_next(request)
