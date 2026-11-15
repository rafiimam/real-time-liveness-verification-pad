"""Auth token issuance and status."""

from fastapi import APIRouter, Depends, HTTPException

from app.security.auth import (
    ADMIN_API_KEY,
    API_KEY,
    AUTH_REQUIRED,
    IssueTokenRequest,
    create_access_token,
    require_admin,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def auth_status():
    return {
        "auth_required": AUTH_REQUIRED,
        "api_key_configured": bool(API_KEY),
        "admin_key_configured": bool(ADMIN_API_KEY),
    }


@router.post("/token/mint")
async def mint_token(body: IssueTokenRequest, _auth: dict = Depends(require_admin)):
    """Mint a short-lived JWT using the admin API key."""
    if body.role not in ("client", "admin"):
        raise HTTPException(status_code=400, detail="role must be client or admin")
    token = create_access_token(
        subject=body.subject,
        role=body.role,
        expires_minutes=body.expires_minutes,
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": body.role,
        "expires_minutes": body.expires_minutes,
    }
