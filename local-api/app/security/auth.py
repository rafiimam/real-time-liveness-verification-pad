"""API key + JWT auth for pilot / production.

When API_KEY is empty and AUTH_REQUIRED=false, auth is skipped (local dev).
Set API_KEY (and optionally JWT_SECRET) before exposing the service.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import Depends, Header, HTTPException, Query, status
from jose import JWTError, jwt
from pydantic import BaseModel

API_KEY = os.getenv("API_KEY", "").strip()
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip() or API_KEY
JWT_SECRET = os.getenv("JWT_SECRET", "").strip() or API_KEY or "dev-insecure-change-me"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
AUTH_REQUIRED = os.getenv("AUTH_REQUIRED", "true" if API_KEY else "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


class IssueTokenRequest(BaseModel):
    subject: str = "client"
    role: str = "client"  # client | admin
    expires_minutes: int | None = None


def create_access_token(
    *,
    subject: str,
    role: str = "client",
    expires_minutes: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    minutes = expires_minutes if expires_minutes is not None else JWT_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_jwt(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def verify_token_string(token: str | None, *, admin: bool = False) -> dict[str, Any]:
    """Validate static API key or JWT. Raises HTTPException on failure."""
    if not AUTH_REQUIRED:
        return {"sub": "anonymous", "role": "admin" if not admin else "admin", "auth": "disabled"}

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials. Pass Authorization: Bearer <API_KEY|JWT>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw = token.strip()
    if raw.lower().startswith("bearer "):
        raw = raw[7:].strip()

    if API_KEY and raw == API_KEY and not admin:
        return {"sub": "api_key", "role": "client", "auth": "api_key"}
    if ADMIN_API_KEY and raw == ADMIN_API_KEY:
        return {"sub": "admin_key", "role": "admin", "auth": "admin_key"}

    try:
        claims = _decode_jwt(raw)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    role = str(claims.get("role") or "client")
    if admin and role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return {**claims, "auth": "jwt"}


def verify_bearer_or_query(
    authorization: str | None = None,
    access_token: str | None = None,
    *,
    admin: bool = False,
) -> dict[str, Any]:
    return verify_token_string(authorization or access_token, admin=admin)


async def require_api_auth(
    authorization: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Query(alias="access_token")] = None,
) -> dict[str, Any]:
    return verify_bearer_or_query(authorization, access_token, admin=False)


async def require_admin(
    authorization: Annotated[str | None, Header()] = None,
    access_token: Annotated[str | None, Query(alias="access_token")] = None,
) -> dict[str, Any]:
    return verify_bearer_or_query(authorization, access_token, admin=True)
