"""Session management endpoints with TTL, nonce binding, and single-use consume."""

from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    CreateSessionRequest,
    CreateSessionResponse,
    SessionStatus,
    VerificationResult,
)
from app.security.auth import require_api_auth

router = APIRouter()

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "120"))
REQUIRE_NONCE = os.getenv("REQUIRE_NONCE", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# In-memory session store (replace with Redis/DB for multi-instance production)
_sessions: dict[str, dict] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def purge_expired_sessions() -> int:
    """Drop expired or fully consumed sessions. Returns count removed."""
    now = _now()
    dead = []
    for sid, sess in _sessions.items():
        expires = sess.get("expires_at")
        if expires is not None and expires < now:
            dead.append(sid)
            continue
        # Keep completed results briefly for polling, then drop
        if sess.get("status") == SessionStatus.COMPLETED and sess.get("consumed_at"):
            consumed = sess["consumed_at"]
            if isinstance(consumed, datetime) and (now - consumed).total_seconds() > 300:
                dead.append(sid)
    for sid in dead:
        _sessions.pop(sid, None)
    return len(dead)


def create_session_record(
    *,
    account_id: str,
    transaction_id: str | None = None,
    transaction_amount: float | None = None,
    status: SessionStatus = SessionStatus.CREATED,
    high_value: bool = False,
) -> dict:
    purge_expired_sessions()
    session_id = str(uuid.uuid4())
    nonce = secrets.token_hex(16)
    now = _now()
    amount = transaction_amount
    if amount is not None and amount >= float(os.getenv("HIGH_VALUE_AMOUNT", "100000")):
        high_value = True
    record = {
        "session_id": session_id,
        "account_id": account_id,
        "transaction_id": transaction_id,
        "transaction_amount": amount,
        "nonce": nonce,
        "status": status,
        "created_at": now,
        "expires_at": now + timedelta(seconds=SESSION_TTL_SECONDS),
        "result": None,
        "consumed": False,
        "consumed_at": None,
        "frame_timestamps": [],
        "high_value": high_value,
        "last_client_ts": None,
    }
    _sessions[session_id] = record
    return record


def get_session(session_id: str) -> dict | None:
    purge_expired_sessions()
    return _sessions.get(session_id)


def update_session(session_id: str, **kwargs):
    if session_id in _sessions:
        _sessions[session_id].update(kwargs)


def ensure_session_active(session_id: str) -> dict:
    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    expires = sess.get("expires_at")
    if expires is not None and expires < _now():
        update_session(session_id, status=SessionStatus.FAILED)
        raise HTTPException(status_code=410, detail="Session expired")
    if sess.get("consumed"):
        raise HTTPException(status_code=409, detail="Session already consumed")
    return sess


def validate_nonce(session_id: str, nonce: str | None) -> None:
    sess = ensure_session_active(session_id)
    if not REQUIRE_NONCE:
        return
    expected = sess.get("nonce")
    if not nonce or not expected or not secrets.compare_digest(str(nonce), str(expected)):
        raise HTTPException(status_code=403, detail="Invalid or missing session nonce")


def record_frame_ts(session_id: str, client_ts: float | int | None) -> bool:
    """Append client timestamp; return False if not monotonic."""
    sess = get_session(session_id)
    if not sess:
        return False
    if client_ts is None:
        return True
    try:
        ts = float(client_ts)
    except (TypeError, ValueError):
        return False
    last = sess.get("last_client_ts")
    stamps = sess.setdefault("frame_timestamps", [])
    stamps.append(ts)
    if last is not None and ts < float(last):
        sess["last_client_ts"] = ts
        return False
    sess["last_client_ts"] = ts
    return True


def consume_session(session_id: str) -> None:
    """Mark session single-use after a verification attempt finishes."""
    if session_id in _sessions:
        _sessions[session_id]["consumed"] = True
        _sessions[session_id]["consumed_at"] = _now()


def timestamps_monotonic(session_id: str) -> bool:
    sess = get_session(session_id)
    if not sess:
        return False
    stamps = sess.get("frame_timestamps") or []
    if len(stamps) < 2:
        return True
    return all(stamps[i] <= stamps[i + 1] for i in range(len(stamps) - 1))


@router.post("/create", response_model=CreateSessionResponse)
async def create_session(
    req: CreateSessionRequest,
    _auth: dict = Depends(require_api_auth),
):
    record = create_session_record(
        account_id=req.account_id,
        transaction_id=req.transaction_id,
        transaction_amount=req.transaction_amount,
    )
    return CreateSessionResponse(
        session_id=record["session_id"],
        nonce=record["nonce"],
        created_at=record["created_at"],
    )


@router.get("/{session_id}/result", response_model=VerificationResult)
async def get_result(
    session_id: str,
    _auth: dict = Depends(require_api_auth),
):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if session["result"]:
        return session["result"]

    return VerificationResult(session_id=session_id, status=session["status"])
