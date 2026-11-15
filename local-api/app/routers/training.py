"""Training data management endpoints (admin only)."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db import training_store
from app.security.auth import require_admin

router = APIRouter()


class LabelRequest(BaseModel):
    label: str  # real | spoof


@router.get("/stats")
async def stats(_auth: dict = Depends(require_admin)):
    return await training_store.training_stats()


@router.post("/sessions/{session_id}/label")
async def relabel(session_id: str, body: LabelRequest, _auth: dict = Depends(require_admin)):
    """Correct auto-label (e.g. live phone wrongly marked spoof)."""
    try:
        ok = await training_store.set_session_label(session_id, body.label)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found in training_sessions")
    return {"session_id": session_id, "label": body.label.lower(), "ok": True}
