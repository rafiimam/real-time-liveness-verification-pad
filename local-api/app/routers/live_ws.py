"""WebSocket live liveness verification for mobile web clients."""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from urllib.parse import parse_qs

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.db.pgvector import store_faceprint
from app.models.schemas import SessionStatus
from app.routers.session import (
    consume_session,
    create_session_record,
    ensure_session_active,
    record_frame_ts,
    update_session,
    validate_nonce,
)
from app.security.auth import AUTH_REQUIRED, verify_token_string
from app.services import gpu_client
from app.services.orchestrator import SKIP_FACE_MATCH, run_verification_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

MIN_FRAMES = 24
MAX_FRAMES = 120
EARLY_FINISH_SECONDS = 4.0
MAX_STREAM_SECONDS = 10.0


def _decode_image(b64: str) -> np.ndarray | None:
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _auth_from_websocket(websocket: WebSocket, msg: dict | None = None) -> dict | None:
    """Validate auth from query string, Authorization header, or message token."""
    if not AUTH_REQUIRED:
        return {"sub": "anonymous", "role": "client", "auth": "disabled"}

    token = None
    qs = parse_qs(websocket.scope.get("query_string", b"").decode())
    if qs.get("access_token"):
        token = qs["access_token"][0]
    if not token:
        headers = {k.decode().lower(): v.decode() for k, v in websocket.scope.get("headers", [])}
        token = headers.get("authorization")
    if not token and msg:
        token = msg.get("access_token") or msg.get("token")
    try:
        return verify_token_string(token, admin=False)
    except Exception as e:
        detail = getattr(e, "detail", str(e))
        raise PermissionError(str(detail)) from e


async def _finish_verification(
    websocket: WebSocket,
    *,
    session_id: str,
    account_id: str,
    frames: list[np.ndarray],
    reference_image: np.ndarray | None,
    nid_photo_b64: str | None = None,
    account_photo_b64: str | None = None,
    customer_name: str | None = None,
    nid_verified: bool = False,
    cbs_faces_detected: int | None = None,
    allow_step_up: bool = True,
) -> bool:
    """Run pipeline and send result. Returns True if a result/error was sent."""
    if len(frames) < MIN_FRAMES:
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        print(f"[WS] too few frames={len(frames)} need>={MIN_FRAMES}", flush=True)
        await websocket.send_json({
            "type": "error",
            "message": f"Too few frames ({len(frames)}). Hold still a bit longer.",
        })
        return True

    video_bytes = b"live-ws-stream"
    print(f"[WS] finishing — pipeline frames={len(frames)}", flush=True)
    try:
        result = await run_verification_pipeline(
            session_id=session_id,
            account_id=account_id,
            frames=frames,
            video_bytes=video_bytes,
            reference_image=reference_image,
            source="websocket_live",
            nid_photo_b64=nid_photo_b64,
            account_photo_b64=account_photo_b64,
            customer_name=customer_name,
            nid_verified=nid_verified,
            cbs_faces_detected=cbs_faces_detected,
            allow_step_up=allow_step_up,
        )
    except Exception as e:
        logger.exception("Live WS verification failed")
        update_session(session_id, status=SessionStatus.FAILED)
        consume_session(session_id)
        await websocket.send_json({
            "type": "error",
            "message": f"Verification failed: {type(e).__name__}: {e}",
        })
        return True

    update_session(session_id, status=SessionStatus.COMPLETED, result=result)
    consume_session(session_id)
    payload = result.model_dump(mode="json")
    payload["type"] = "result"
    payload["frames_received"] = len(frames)
    payload["source"] = "websocket_live"
    print(f"[WS] result approved={payload.get('is_approved')} frames={len(frames)}", flush=True)
    await websocket.send_json(payload)
    return True


@router.websocket("/ws/live")
async def live_verify(websocket: WebSocket):
    """Real-time mobile stream protocol.

    Client messages (JSON):
      {"type":"start","account_number":"123…","photo_base64":"<jpeg b64>","access_token":"..."}
      {"type":"frame","image_base64":"<jpeg b64>","ts": 123456,"nonce":"..."}
      {"type":"end","nonce":"..."}

    `account_number` preferred (SDK); `account_id` still accepted for web clients.
    Auth: query ?access_token=, Authorization header, or start.access_token
    """
    await websocket.accept()
    session_id = None
    account_id = "MOBILE_USER"
    session_nonce: str | None = None
    frames: list[np.ndarray] = []
    reference_image: np.ndarray | None = None
    started_at: float | None = None
    finishing = False
    nid_photo_b64: str | None = None
    account_photo_b64: str | None = None
    customer_name: str | None = None
    nid_verified: bool = False
    cbs_faces_detected: int | None = None
    allow_step_up: bool = True

    try:
        # Early auth check from query/header so attackers don't stream freely
        try:
            _auth_from_websocket(websocket)
        except PermissionError as e:
            await websocket.send_json({"type": "error", "message": f"Unauthorized: {e}"})
            await websocket.close(code=4401)
            return

        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            print(f"[WS] recv type={mtype} session={session_id} frames={len(frames)}", flush=True)

            if finishing:
                continue

            if mtype == "start":
                try:
                    _auth_from_websocket(websocket, msg)
                except PermissionError as e:
                    await websocket.send_json({"type": "error", "message": f"Unauthorized: {e}"})
                    continue

                # Prefer account_number (SDK / bank apps); account_id kept for web compatibility.
                # Lookup / KYC bind against account_number will be added later on the backend.
                raw_acct = (
                    msg.get("account_number")
                    or msg.get("accountNumber")
                    or msg.get("account_id")
                    or "MOBILE_USER"
                )
                from app.services.account_lookup import resolve_account_number

                resolved = await resolve_account_number(str(raw_acct))
                account_id = str(resolved.get("subject_id") or raw_acct).strip() or "MOBILE_USER"
                nid_photo_b64 = resolved.get("nid_photo_b64")
                account_photo_b64 = resolved.get("account_photo_b64")
                customer_name = resolved.get("customer_name")
                nid_verified = resolved.get("nid_verified", False)
                if resolved.get("errors"):
                    logger.info("Account resolve partial errors: %s", resolved["errors"])

                from app.services.training_capture import show_bank_reference_photos

                show_bank_reference_photos(
                    nid_photo_b64=nid_photo_b64,
                    account_photo_b64=account_photo_b64,
                )

                photo_b64 = msg.get("photo_base64") or ""
                if not photo_b64:
                    await websocket.send_json({"type": "error", "message": "photo_base64 required"})
                    continue

                image = _decode_image(photo_b64)
                if image is None:
                    await websocket.send_json({"type": "error", "message": "Invalid reference photo"})
                    continue
                reference_image = image
                print(
                    f"[WS] reference photo decoded shape={image.shape} "
                    f"account_number={resolved.get('account_number')} subject={account_id}",
                    flush=True,
                )

                embedding = await gpu_client.get_embedding(image)
                if embedding is None:
                    if SKIP_FACE_MATCH:
                        embedding = [0.0] * 512
                        logger.warning("No face in reference photo — continuing (SKIP_FACE_MATCH)")
                        print("[WS] No face in reference — continuing (SKIP_FACE_MATCH)", flush=True)
                    else:
                        await websocket.send_json({"type": "error", "message": "No face in reference photo"})
                        continue
                else:
                    print(f"[WS] reference embedding dim={len(embedding)}", flush=True)

                clean_b64 = photo_b64.split(",")[-1]
                await store_faceprint(
                    account_id,
                    embedding,
                    reference_image_b64=clean_b64,
                    source="client_photo",
                    face_index=0,
                    provider_code="internal",
                )

                # Extract multi-face templates from CBS signature card + NID
                from app.services.internal_facematch import prepare_account_templates

                tmpl = await prepare_account_templates(
                    account_id,
                    nid_photo_b64=nid_photo_b64,
                    account_photo_b64=account_photo_b64,
                )
                cbs_faces_detected = int(tmpl.get("cbs_faces_detected") or 0)
                if tmpl.get("errors"):
                    logger.info("Template prep messages: %s", tmpl["errors"])
                print(
                    f"[WS] templates cbs={cbs_faces_detected} "
                    f"nid={tmpl.get('nid_faces_detected')} errors={tmpl.get('errors')}",
                    flush=True,
                )

                amount = msg.get("transaction_amount")
                try:
                    amount_f = float(amount) if amount is not None else None
                except (TypeError, ValueError):
                    amount_f = None
                record = create_session_record(
                    account_id=account_id,
                    transaction_id=msg.get("transaction_id"),
                    transaction_amount=amount_f,
                    status=SessionStatus.PROCESSING,
                )
                session_id = record["session_id"]
                session_nonce = record["nonce"]
                frames = []
                finishing = False
                # Second challenge round: client sets step_up_completed / challenge_round
                allow_step_up = not bool(
                    msg.get("step_up_completed")
                    or msg.get("challenge_round")
                    or msg.get("is_step_up")
                )
                started_at = datetime.now(timezone.utc).timestamp()
                print(f"[WS] session ready id={session_id} allow_step_up={allow_step_up}", flush=True)
                await websocket.send_json({
                    "type": "ready",
                    "session_id": session_id,
                    "nonce": session_nonce,
                    "early_seconds": EARLY_FINISH_SECONDS,
                    "max_seconds": MAX_STREAM_SECONDS,
                    "min_frames": MIN_FRAMES,
                    "expires_at": record["expires_at"].isoformat(),
                })

            elif mtype == "frame":
                if not session_id:
                    await websocket.send_json({"type": "error", "message": "Call start first"})
                    continue
                try:
                    ensure_session_active(session_id)
                    validate_nonce(session_id, msg.get("nonce"))
                except Exception as e:
                    detail = getattr(e, "detail", str(e))
                    await websocket.send_json({"type": "error", "message": str(detail)})
                    continue
                if len(frames) >= MAX_FRAMES:
                    continue

                if not record_frame_ts(session_id, msg.get("ts")):
                    print("[WS] non-monotonic ts — rejected frame", flush=True)
                    await websocket.send_json({
                        "type": "error",
                        "message": "Frame timestamps must be monotonic",
                    })
                    continue

                frame = _decode_image(msg.get("image_base64") or "")
                if frame is None:
                    print("[WS] frame decode failed — skipped", flush=True)
                    continue
                frames.append(frame)
                if len(frames) == 1 or len(frames) % 10 == 0:
                    print(f"[WS] frames received={len(frames)} last_shape={frame.shape}", flush=True)

                elapsed_ms = 0
                if started_at is not None:
                    elapsed_ms = int((datetime.now(timezone.utc).timestamp() - started_at) * 1000)
                elapsed_s = elapsed_ms / 1000.0
                can_finish = len(frames) >= MIN_FRAMES and elapsed_s >= EARLY_FINISH_SECONDS
                await websocket.send_json({
                    "type": "progress",
                    "frames": len(frames),
                    "elapsed_ms": elapsed_ms,
                    "can_finish": can_finish,
                })

                if can_finish or (len(frames) >= MIN_FRAMES and elapsed_s >= MAX_STREAM_SECONDS):
                    finishing = True
                    await websocket.send_json({"type": "status", "message": "Checking…"})
                    await _finish_verification(
                        websocket,
                        session_id=session_id,
                        account_id=account_id,
                        frames=list(frames),
                        reference_image=reference_image,
                        nid_photo_b64=nid_photo_b64,
                        account_photo_b64=account_photo_b64,
                        customer_name=customer_name,
                        nid_verified=nid_verified,
                        cbs_faces_detected=cbs_faces_detected,
                        allow_step_up=allow_step_up,
                    )
                    session_id = None
                    session_nonce = None
                    frames = []
                    reference_image = None
                    finishing = False

            elif mtype == "end":
                if not session_id:
                    await websocket.send_json({"type": "error", "message": "No active session"})
                    continue
                try:
                    ensure_session_active(session_id)
                    validate_nonce(session_id, msg.get("nonce"))
                except Exception as e:
                    detail = getattr(e, "detail", str(e))
                    await websocket.send_json({"type": "error", "message": str(detail)})
                    continue
                finishing = True
                await _finish_verification(
                    websocket,
                    session_id=session_id,
                    account_id=account_id,
                    frames=list(frames),
                    reference_image=reference_image,
                    nid_photo_b64=nid_photo_b64,
                    account_photo_b64=account_photo_b64,
                    customer_name=customer_name,
                    nid_verified=nid_verified,
                    cbs_faces_detected=cbs_faces_detected,
                    allow_step_up=allow_step_up,
                )
                session_id = None
                session_nonce = None
                frames = []
                reference_image = None
                finishing = False

            elif mtype == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {mtype}"})

    except WebSocketDisconnect:
        logger.info("Live WS client disconnected session=%s", session_id)
        print(f"[WS] disconnect session={session_id}", flush=True)
        if session_id:
            consume_session(session_id)
    except Exception as e:
        logger.exception("Live WS error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        if session_id:
            consume_session(session_id)
