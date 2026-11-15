"""Admin APIs for face-match provider switching."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import pgvector
from app.security.auth import require_admin

router = APIRouter(prefix="/admin/face-match", tags=["admin-face-match"])


class ActivateRequest(BaseModel):
    provider_id: int | None = None


@router.get("/providers")
async def list_providers(_auth: dict = Depends(require_admin)):
    providers = await pgvector.list_face_match_providers()
    active = await pgvector.get_active_face_match_provider()
    return {
        "providers": providers,
        "active": active,
    }


@router.post("/providers/{provider_id}/activate")
async def activate_provider(provider_id: int, _auth: dict = Depends(require_admin)):
    try:
        provider = await pgvector.set_active_face_match_provider(provider_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "active": provider,
        "message": f"Active face-match provider set to {provider.get('code')} (id={provider_id})",
    }
