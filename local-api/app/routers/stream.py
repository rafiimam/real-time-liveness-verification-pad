"""Stream upload endpoint — receives 10s video and triggers processing."""

import logging
import os
import tempfile

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.models.schemas import SessionStatus
from app.routers.session import consume_session, ensure_session_active, update_session
from app.security.auth import require_api_auth
from app.services.orchestrator import run_verification_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOW_VERIFY_UPLOAD = os.getenv("ALLOW_VERIFY_UPLOAD", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@router.post("/{session_id}/stream")
async def upload_stream(
    session_id: str,
    video: UploadFile = File(...),
    _auth: dict = Depends(require_api_auth),
):
    if not ALLOW_VERIFY_UPLOAD:
        raise HTTPException(
            status_code=403,
            detail="Upload stream disabled. Use live WebSocket /ws/live.",
        )

    session = ensure_session_active(session_id)
    if session["status"] != SessionStatus.CREATED:
        raise HTTPException(status_code=400, detail="Session already processed")

    update_session(session_id, status=SessionStatus.PROCESSING)

    content = await video.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        raise HTTPException(status_code=400, detail="No frames extracted from video")

    try:
        result = await run_verification_pipeline(
            session_id=session_id,
            account_id=session["account_id"],
            frames=frames,
            video_bytes=content,
            source="session_stream",
        )
    except Exception as e:
        logger.exception("Verification pipeline failed for session %s", session_id)
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        raise HTTPException(status_code=500, detail=f"Verification failed: {type(e).__name__}: {e}") from e

    update_session(session_id, status=SessionStatus.COMPLETED, result=result)
    consume_session(session_id)
    return result
