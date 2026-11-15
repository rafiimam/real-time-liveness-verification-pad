"""Capture verification runs into Postgres for continuous training."""

from __future__ import annotations

import base64
import logging
import os

import cv2
import numpy as np

from app.db import training_store
from app.services.debug_images import print_image_console

logger = logging.getLogger(__name__)

SAVE_TRAINING_DATA = os.getenv("SAVE_TRAINING_DATA", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
SCREEN_SPOOF_THRESHOLD = float(os.getenv("SCREEN_SPOOF_THRESHOLD", "0.46"))


def decode_image_b64(b64: str | None) -> np.ndarray | None:
    """Decode a data-URL or raw base64 JPEG/PNG to BGR."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(str(b64).split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        logger.exception("Failed to decode image b64")
        return None


def show_bank_reference_photos(
    *,
    nid_photo_b64: str | None = None,
    account_photo_b64: str | None = None,
    nid_image: np.ndarray | None = None,
    cbs_image: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Console-log NID + CBS/signature photos; return decoded BGR arrays."""
    nid = nid_image if nid_image is not None else decode_image_b64(nid_photo_b64)
    cbs = cbs_image if cbs_image is not None else decode_image_b64(account_photo_b64)

    if nid is not None:
        print(f"[BANK PHOTO] NID photo shape={nid.shape}", flush=True)
        print_image_console(nid, title="NID_PHOTO")
    else:
        print("[BANK PHOTO] NID photo: <none>", flush=True)

    if cbs is not None:
        print(f"[BANK PHOTO] CBS / signature card shape={cbs.shape}", flush=True)
        print_image_console(cbs, title="CBS_SIGNATURE_CARD")
    else:
        print("[BANK PHOTO] CBS / signature card: <none>", flush=True)

    return nid, cbs


def save_face_match_frame(
    session_id: str,
    frame: np.ndarray,
    blur_score: float,
    frame_index: int | None = None,
) -> None:
    """Log + visually print the frame chosen for face matching (no local folder)."""
    print(
        f"[FACE MATCH] session={session_id} index={frame_index} blur={blur_score:.2f}",
        flush=True,
    )
    print_image_console(frame, title=f"FACE_MATCH idx={frame_index} blur={blur_score:.1f}")
    logger.info(
        "FACE_MATCH_FRAME session=%s index=%s blur=%.2f",
        session_id,
        frame_index,
        blur_score,
    )


async def capture_training_session(
    *,
    session_id: str,
    account_id: str,
    frames: list[np.ndarray],
    reference_image: np.ndarray | None,
    face_match_frame: np.ndarray | None,
    face_match_meta: dict,
    result: dict,
    source: str,
    nid_photo_b64: str | None = None,
    account_photo_b64: str | None = None,
    nid_image: np.ndarray | None = None,
    cbs_image: np.ndarray | None = None,
) -> dict:
    if not SAVE_TRAINING_DATA:
        logger.info("SAVE_TRAINING_DATA=false — skip")
        return {}

    screen_score = float(result.get("screen_spoof_score") or 0.0)
    is_screen = screen_score >= SCREEN_SPOOF_THRESHOLD
    approved = bool(result.get("is_approved"))
    # Prefer explicit screen-replay cue for spoof labels. Do NOT trust spatial UAD
    # alone — phone-domain scores are noisy and poisoned prior finetunes.
    if is_screen:
        auto_label = "spoof"
    elif approved:
        auto_label = "real"
    else:
        # Live phone reject for non-screen reasons → still treat as real for training
        auto_label = "real"

    temporal_label = "fake" if auto_label == "spoof" else "real"

    print("\n========== TRAINING CAPTURE (DATABASE) ==========", flush=True)
    print(
        f"[TRAINING] session={session_id} auto_label={auto_label} "
        f"screen={screen_score:.3f} approved={approved}",
        flush=True,
    )

    if reference_image is not None:
        print_image_console(reference_image, title="REFERENCE_PHOTO")
    if face_match_frame is not None:
        print_image_console(face_match_frame, title="FACE_MATCH_FRAME")

    nid_img, cbs_img = show_bank_reference_photos(
        nid_photo_b64=nid_photo_b64,
        account_photo_b64=account_photo_b64,
        nid_image=nid_image,
        cbs_image=cbs_image,
    )

    pk = await training_store.store_training_session(
        session_id=session_id,
        account_id=account_id,
        source=source,
        frames=frames,
        reference_image=reference_image,
        face_match_frame=face_match_frame,
        face_match_meta=face_match_meta,
        result=result,
        auto_label=auto_label,
        temporal_label=temporal_label,
        nid_image=nid_img,
        cbs_image=cbs_img,
    )

    stats = await training_store.training_stats()
    print(f"[TRAINING-DB] stats={stats}", flush=True)
    print("========== END TRAINING CAPTURE ==========\n", flush=True)

    return {
        "session_pk": pk,
        "auto_label": auto_label,
        "temporal_label": temporal_label,
        "stats": stats,
        "stored_nid": nid_img is not None,
        "stored_cbs": cbs_img is not None,
    }
