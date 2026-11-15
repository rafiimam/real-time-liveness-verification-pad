"""Concurrent pipeline orchestrator.

Runs injection check, spatial liveness, temporal liveness, screen-replay
heuristics, and face match, then aggregates into a composite decision.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import numpy as np

from app.models.schemas import SessionStatus, VerificationResult
from app.services import gpu_client, quality, screen_replay, training_capture, internal_facematch
from app.db.pgvector import (
    get_active_face_match_provider,
    get_reference_image_b64,
    list_faceprints_for_account,
)
from app.db.audit import log_verification
from app.services.facematch import AZURE_MATCH_THRESHOLD, match_faces_routed

logger = logging.getLogger(__name__)

# Thresholds (overridable via env after calibration)
IDENTITY_THRESHOLD = 0.85
SPATIAL_LIVENESS_THRESHOLD = float(os.getenv("SPATIAL_LIVENESS_THRESHOLD", "0.5"))
TEMPORAL_LIVENESS_THRESHOLD = float(os.getenv("TEMPORAL_LIVENESS_THRESHOLD", "0.90"))
REQUIRE_TEMPORAL = os.getenv("REQUIRE_TEMPORAL", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
REQUIRE_SPATIAL = os.getenv("REQUIRE_SPATIAL", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
# Banking PAD: every modality has a hard floor. Strong temporal/screen must NOT
# waive a failed spatial check (masks / texture / deepfake risk).
SPATIAL_MIN_PASS_FRACTION = float(os.getenv("SPATIAL_MIN_PASS_FRACTION", "0.6"))
LIVENESS_WEIGHT_SPATIAL = float(os.getenv("LIVENESS_WEIGHT_SPATIAL", "0.40"))
LIVENESS_WEIGHT_TEMPORAL = float(os.getenv("LIVENESS_WEIGHT_TEMPORAL", "0.35"))
LIVENESS_WEIGHT_SCREEN = float(os.getenv("LIVENESS_WEIGHT_SCREEN", "0.25"))
COMPOSITE_LIVENESS_THRESHOLD = float(os.getenv("COMPOSITE_LIVENESS_THRESHOLD", "0.55"))
QUALITY_GATE_ENABLED = os.getenv("QUALITY_GATE_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
QUALITY_MIN_BLUR = float(os.getenv("QUALITY_MIN_BLUR", "80"))
QUALITY_MIN_BRIGHTNESS = float(os.getenv("QUALITY_MIN_BRIGHTNESS", "40"))
QUALITY_MAX_BRIGHTNESS = float(os.getenv("QUALITY_MAX_BRIGHTNESS", "220"))
# Borderline spatial → step-up challenge (never soft-pass a hard spatial fail)
SPATIAL_BORDERLINE_LOW = float(os.getenv("SPATIAL_BORDERLINE_LOW", "0.45"))
SPATIAL_BORDERLINE_HIGH = float(os.getenv("SPATIAL_BORDERLINE_HIGH", "0.70"))
SPATIAL_BORDERLINE_PASS_FRAC_LOW = float(os.getenv("SPATIAL_BORDERLINE_PASS_FRAC_LOW", "0.30"))
STEP_UP_MIN_TEMPORAL = float(os.getenv("STEP_UP_MIN_TEMPORAL", "0.85"))
STEP_UP_ENABLED = os.getenv("STEP_UP_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
SKIP_FACE_MATCH = os.getenv("SKIP_FACE_MATCH", "false").strip().lower() in (
    "1", "true", "yes", "on",
)

_CHALLENGE_PROMPTS = {
    "turn_left": "Slowly turn your head left, then face the camera again",
    "turn_right": "Slowly turn your head right, then face the camera again",
    "move_closer_hold": "Move slightly closer and hold still for a few seconds",
}


def _pick_challenge() -> dict:
    import random

    code = random.choice(list(_CHALLENGE_PROMPTS.keys()))
    return {"code": code, "prompt": _CHALLENGE_PROMPTS[code]}


def _spatial_is_borderline(spatial_result: dict) -> bool:
    spoof = float(spatial_result.get("mean_spoof_score", 1.0))
    pass_frac = float(spatial_result.get("pass_fraction", 0.0))
    in_spoof_band = SPATIAL_BORDERLINE_LOW <= spoof < SPATIAL_BORDERLINE_HIGH
    in_frac_band = SPATIAL_BORDERLINE_PASS_FRAC_LOW <= pass_frac < SPATIAL_MIN_PASS_FRACTION
    return in_spoof_band or in_frac_band


def _select_face_match_frame(frames: list[np.ndarray]) -> dict:
    """Pick sharpest frame for face match; always log details to console."""
    if not frames:
        print("[FACE MATCH] No frames available", flush=True)
        logger.warning("FACE_MATCH: no frames")
        return {"frame": None, "frame_index": None, "blur_score": 0.0}

    scored = [(i, quality.laplacian_variance(f), f) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[1], reverse=True)
    best_i, best_blur, best_f = scored[0]
    top5 = [(i, round(b, 2)) for i, b, _ in scored[:5]]
    print(
        f"[FACE MATCH] candidates top5 (index, blur)={top5} "
        f"-> selected index={best_i} blur={best_blur:.2f} shape={best_f.shape}",
        flush=True,
    )
    logger.info(
        "FACE_MATCH select index=%s blur=%.2f shape=%s top5=%s",
        best_i,
        best_blur,
        best_f.shape,
        top5,
    )
    return {"frame": best_f, "frame_index": best_i, "blur_score": float(best_blur)}


async def _injection_check(
    video_bytes: bytes,
    frames: list[np.ndarray],
    *,
    session_id: str | None = None,
) -> dict:
    live_ws = video_bytes == b"live-ws-stream"
    # Align with live_ws early-finish (~24 frames @ ~8fps). Upload path still expects denser clips.
    min_frames = 24 if live_ws else 30
    max_frames = 150 if live_ws else 600

    ts_ok = True
    if session_id and live_ws:
        from app.routers.session import timestamps_monotonic

        ts_ok = timestamps_monotonic(session_id)

    checks = {
        "frame_count_valid": min_frames <= len(frames) <= max_frames,
        "consistent_resolution": True,
        "timestamp_monotonic": ts_ok,
        "live_ws": live_ws,
        "frame_count": len(frames),
        "min_frames": min_frames,
        "max_frames": max_frames,
    }

    if frames:
        h, w = frames[0].shape[:2]
        checks["resolution"] = f"{w}x{h}"
        for f in frames[1:]:
            if f.shape[0] != h or f.shape[1] != w:
                checks["consistent_resolution"] = False
                break

    checks["passed"] = all(
        v for k, v in checks.items()
        if k not in ("live_ws", "frame_count", "min_frames", "max_frames", "resolution")
    )
    print(f"[INJECTION] {checks}", flush=True)
    logger.info("INJECTION %s", checks)
    return checks


async def _face_match_against_b64(
    frames: list[np.ndarray],
    reference_b64: str,
    match_meta: dict,
    label: str,
    *,
    account_id: str | None = None,
) -> dict:
    """Match best frame against a specific base64 reference image."""
    best_frame = match_meta.get("frame")
    if best_frame is None:
        print(f"[{label}] FAIL — no usable frame", flush=True)
        return {"is_same_person": False, "confidence_score": 0.0}
    if not reference_b64:
        print(f"[{label}] FAIL — no reference image", flush=True)
        return {"is_same_person": False, "confidence_score": 0.0, "error": "no_reference"}

    print(f"[{label}] calling match APIs frame_index={match_meta.get('frame_index')}", flush=True)
    result = await match_faces_routed(
        best_frame, reference_b64, account_id=account_id
    )
    print(f"[{label}] result={result}", flush=True)
    logger.info("%s result=%s", label, result)
    return result


async def _face_match(frames: list[np.ndarray], account_id: str, match_meta: dict) -> dict:
    if SKIP_FACE_MATCH:
        print(
            f"[FACE MATCH] SKIP_FACE_MATCH=true — not calling APIs "
            f"(selected frame index={match_meta.get('frame_index')} still saved for training)",
            flush=True,
        )
        return {
            "is_same_person": True,
            "confidence_score": 1.0,
            "source": "skipped",
            "frame_index": match_meta.get("frame_index"),
            "blur_score": match_meta.get("blur_score"),
        }

    best_frame = match_meta.get("frame")
    if best_frame is None:
        print("[FACE MATCH] FAIL — no usable frame", flush=True)
        return {"is_same_person": False, "confidence_score": 0.0}

    reference_b64 = await get_reference_image_b64(account_id)
    templates = await list_faceprints_for_account(account_id)
    if reference_b64 is None and not templates:
        print(f"[FACE MATCH] FAIL — no reference for account={account_id}", flush=True)
        return {"is_same_person": False, "confidence_score": 0.0, "error": "no_reference"}

    print(
        f"[FACE MATCH] calling match APIs account={account_id} "
        f"frame_index={match_meta.get('frame_index')} blur={match_meta.get('blur_score'):.2f} "
        f"templates={len(templates)}",
        flush=True,
    )
    result = await match_faces_routed(
        best_frame, reference_b64, account_id=account_id
    )
    print(f"[FACE MATCH] result={result}", flush=True)
    logger.info("FACE_MATCH result=%s", result)
    result["frame_index"] = match_meta.get("frame_index")
    result["blur_score"] = match_meta.get("blur_score")
    return result


async def _internal_source_matches(
    best_frame: np.ndarray,
    account_id: str,
    *,
    has_nid: bool,
    has_cbs: bool,
) -> tuple[dict, dict]:
    """Template-based NID / CBS matches (any-of joint-account)."""
    split = await internal_facematch.match_live_split_by_source(best_frame, account_id)
    nid_r = split["nid"] if has_nid else {"is_same_person": False, "confidence_score": 0.0, "error": "no_nid"}
    cbs_r = split["cbs"] if has_cbs else {"is_same_person": False, "confidence_score": 0.0, "error": "no_cbs"}
    if has_nid and nid_r.get("error") == "no_templates":
        nid_r = {"is_same_person": False, "confidence_score": 0.0, "error": "no_templates"}
    if has_cbs and cbs_r.get("error") == "no_templates":
        cbs_r = {"is_same_person": False, "confidence_score": 0.0, "error": "no_templates"}
    return nid_r, cbs_r


async def _spatial_liveness(frames: list[np.ndarray]) -> dict:
    sampled = quality.select_spatial_samples(frames, max_samples=10)
    print(
        f"[SPATIAL] input_frames={len(frames)} sampled={len(sampled)} "
        f"threshold={SPATIAL_LIVENESS_THRESHOLD} require={REQUIRE_SPATIAL}",
        flush=True,
    )
    if not sampled:
        print("[SPATIAL] FAIL — no usable quality frames", flush=True)
        return {"mean_spoof_score": 1.0, "is_live": False, "frames_analyzed": 0}

    scores = []
    for i, frame in enumerate(sampled):
        score = await gpu_client.get_spoof_score(frame)
        scores.append(score)
        print(f"[SPATIAL] sample[{i}] spoof_score={score:.6f}", flush=True)

    mean_score = float(sum(scores) / len(scores))
    median_score = float(np.median(scores))
    # Median resists single-frame spikes; majority vote resists intermittent spoof frames.
    decision_score = median_score
    pass_count = sum(1 for s in scores if s < SPATIAL_LIVENESS_THRESHOLD)
    pass_fraction = pass_count / len(scores)
    is_live = (
        decision_score < SPATIAL_LIVENESS_THRESHOLD
        and pass_fraction >= SPATIAL_MIN_PASS_FRACTION
    )
    print(
        f"[SPATIAL] mean_spoof={mean_score:.6f} median_spoof={median_score:.6f} "
        f"decision={decision_score:.6f} pass_frac={pass_fraction:.2f} "
        f"(min={SPATIAL_MIN_PASS_FRACTION}) is_live={is_live} analyzed={len(sampled)}",
        flush=True,
    )
    logger.info(
        "SPATIAL mean=%.6f median=%.6f pass_frac=%.2f is_live=%s scores=%s",
        mean_score,
        median_score,
        pass_fraction,
        is_live,
        [round(s, 6) for s in scores],
    )
    return {
        "mean_spoof_score": decision_score,
        "mean_spoof_raw": mean_score,
        "median_spoof_score": median_score,
        "pass_fraction": pass_fraction,
        "is_live": is_live,
        "frames_analyzed": len(sampled),
        "scores": scores,
    }


async def _temporal_liveness(frames: list[np.ndarray]) -> dict:
    sharp_frames = quality.filter_sharp_frames(frames)
    print(
        f"[TEMPORAL] input_frames={len(frames)} sharp={len(sharp_frames)} "
        f"threshold={TEMPORAL_LIVENESS_THRESHOLD} require={REQUIRE_TEMPORAL}",
        flush=True,
    )
    if not sharp_frames:
        print("[TEMPORAL] FAIL — no sharp frames", flush=True)
        return {"liveness_score": 0.0, "is_live": False}

    step = max(1, len(sharp_frames) // 60)
    sampled = sharp_frames[::step][:60]
    print(f"[TEMPORAL] sending {len(sampled)} frames to GPU", flush=True)

    result = await gpu_client.get_temporal_liveness(sampled)
    liveness_score = result["liveness_score"]
    is_live = liveness_score > TEMPORAL_LIVENESS_THRESHOLD
    print(
        f"[TEMPORAL] liveness={liveness_score:.6f} is_live={is_live} "
        f"chunks={result.get('chunk_scores')}",
        flush=True,
    )
    logger.info("TEMPORAL result=%s", result)
    return {
        "liveness_score": liveness_score,
        "is_live": is_live,
        "chunk_scores": result.get("chunk_scores", []),
    }


async def _screen_replay_check(frames: list[np.ndarray]) -> dict:
    result = await asyncio.to_thread(screen_replay.analyze_screen_replay, frames)
    print(
        f"[SCREEN] score={result.get('screen_spoof_score'):.4f} "
        f"is_replay={result.get('is_screen_replay')} "
        f"components={result.get('components')} strong={result.get('strong_cues')}",
        flush=True,
    )
    logger.info("SCREEN %s", result)
    return result


async def run_verification_pipeline(
    session_id: str,
    account_id: str,
    frames: list[np.ndarray],
    video_bytes: bytes,
    reference_image: np.ndarray | None = None,
    source: str = "verify",
    nid_photo_b64: str | None = None,
    account_photo_b64: str | None = None,
    customer_name: str | None = None,
    nid_verified: bool = False,
    cbs_faces_detected: int | None = None,
    allow_step_up: bool = True,
) -> VerificationResult:
    """Execute all verification layers concurrently and produce final decision."""

    print(
        f"\n########## VERIFY START session={session_id} account={account_id} "
        f"frames={len(frames)} source={source} step_up_allowed={allow_step_up} ##########",
        flush=True,
    )
    logger.info(
        "VERIFY START session=%s account=%s frames=%s source=%s",
        session_id,
        account_id,
        len(frames),
        source,
    )

    match_meta = _select_face_match_frame(frames)
    if match_meta.get("frame") is not None:
        training_capture.save_face_match_frame(
            session_id,
            match_meta["frame"],
            match_meta["blur_score"],
            match_meta.get("frame_index"),
        )

    capture_trust = "live_ws" if video_bytes == b"live-ws-stream" else "upload"
    if source:
        if "websocket" in source:
            capture_trust = "live_ws"
        elif "upload" in source or "verify" in source:
            capture_trust = "upload"

    # --- Strict quality gate BEFORE PAD (fail-fast retry) ---
    quality_info = quality.assess_capture_quality(
        frames,
        min_blur=QUALITY_MIN_BLUR,
        min_brightness=QUALITY_MIN_BRIGHTNESS,
        max_brightness=QUALITY_MAX_BRIGHTNESS,
    )
    quality_adequate = bool(quality_info.get("adequate", True))
    if QUALITY_GATE_ENABLED and not quality_adequate:
        reason_codes = ["CAPTURE_QUALITY"] + [
            f"QUALITY_{r}" for r in (quality_info.get("reasons") or [])
        ]
        print(
            f"[QUALITY] PREFLIGHT FAIL — skipping PAD reasons={reason_codes} "
            f"blur={quality_info.get('median_blur')} brightness={quality_info.get('mean_brightness')}",
            flush=True,
        )
        result = VerificationResult(
            session_id=session_id,
            status=SessionStatus.COMPLETED,
            face_match=None,
            face_match_confidence=None,
            nid_photo_match=None,
            account_photo_match=None,
            nid_verified=nid_verified if nid_photo_b64 else None,
            customer_name=customer_name,
            cbs_faces_detected=cbs_faces_detected,
            liveness=False,
            spatial_liveness_score=None,
            temporal_liveness_score=None,
            screen_spoof_score=None,
            composite_risk_score=0.75,
            is_approved=False,
            reason_codes=reason_codes,
            capture_trust=capture_trust,
            challenge=None,
            completed_at=datetime.now(timezone.utc),
        )
        result_dict = result.model_dump(mode="json")
        await log_verification(session_id, account_id, result_dict)
        try:
            await training_capture.capture_training_session(
                session_id=session_id,
                account_id=account_id,
                frames=frames,
                reference_image=reference_image,
                face_match_frame=match_meta.get("frame"),
                face_match_meta={
                    "frame_index": match_meta.get("frame_index"),
                    "blur_score": match_meta.get("blur_score"),
                    "quality_preflight": quality_info,
                },
                result=result_dict,
                source=source,
                nid_photo_b64=nid_photo_b64,
                account_photo_b64=account_photo_b64,
            )
        except Exception:
            logger.exception("Training capture failed session=%s", session_id)
        print(
            f"########## VERIFY END session={session_id} approved=False "
            f"(CAPTURE_QUALITY) ##########\n",
            flush=True,
        )
        return result

    injection_task = asyncio.create_task(
        _injection_check(video_bytes, frames, session_id=session_id)
    )
    spatial_task = asyncio.create_task(_spatial_liveness(frames))
    temporal_task = asyncio.create_task(_temporal_liveness(frames))
    screen_task = asyncio.create_task(_screen_replay_check(frames))

    # Multi-reference face matching: NID photo + account photo + stored faceprint
    nid_match_result: dict = {"is_same_person": False, "confidence_score": 0.0}
    acct_match_result: dict = {"is_same_person": False, "confidence_score": 0.0}
    provider_used: str | None = None
    active_provider = await get_active_face_match_provider()
    active_code = (active_provider.get("code") or "internal").lower()

    if SKIP_FACE_MATCH:
        match_result = await _face_match(frames, account_id, match_meta)
        injection_result, spatial_result, temporal_result, screen_result = await asyncio.gather(
            injection_task, spatial_task, temporal_task, screen_task
        )
    else:
        match_task = asyncio.create_task(_face_match(frames, account_id, match_meta))
        best_frame = match_meta.get("frame")

        # Prefer template-based NID/CBS when internal is active (multi-face joint accounts)
        use_templates = active_code == "internal" and best_frame is not None
        nid_match_task = None
        acct_match_task = None

        if use_templates and (nid_photo_b64 or account_photo_b64 or True):
            templates = await list_faceprints_for_account(account_id)
            has_nid_t = any(t.get("source") == "nid" for t in templates)
            has_cbs_t = any(t.get("source") == "cbs_sign" for t in templates)
            if has_nid_t or has_cbs_t:
                nid_match_result, acct_match_result = await _internal_source_matches(
                    best_frame,
                    account_id,
                    has_nid=has_nid_t or bool(nid_photo_b64),
                    has_cbs=has_cbs_t or bool(account_photo_b64),
                )
                if cbs_faces_detected is None:
                    cbs_faces_detected = sum(
                        1 for t in templates if t.get("source") == "cbs_sign"
                    )
            else:
                use_templates = False

        if not use_templates:
            if nid_photo_b64:
                nid_match_task = asyncio.create_task(
                    _face_match_against_b64(
                        frames, nid_photo_b64, match_meta, "NID_MATCH",
                        account_id=account_id,
                    )
                )
            if account_photo_b64:
                acct_match_task = asyncio.create_task(
                    _face_match_against_b64(
                        frames, account_photo_b64, match_meta, "ACCT_MATCH",
                        account_id=account_id,
                    )
                )

        gather_list = [injection_task, spatial_task, temporal_task, screen_task, match_task]
        if nid_match_task:
            gather_list.append(nid_match_task)
        if acct_match_task:
            gather_list.append(acct_match_task)

        gathered = await asyncio.gather(*gather_list)
        injection_result = gathered[0]
        spatial_result = gathered[1]
        temporal_result = gathered[2]
        screen_result = gathered[3]
        match_result = gathered[4]
        gi = 5
        if nid_match_task:
            nid_match_result = gathered[gi]
            gi += 1
        if acct_match_task:
            acct_match_result = gathered[gi]

    # Continue with identity / PAD decision
    provider_used = (
        match_result.get("provider_used")
        or match_result.get("source")
        or active_code
    )
    face_matched = match_result.get("is_same_person", False)
    face_confidence = match_result.get("confidence_score", 0.0)
    match_source = match_result.get("source", "external_api")
    spatial_live = spatial_result.get("is_live", False)
    temporal_live = temporal_result.get("is_live", False)
    injection_passed = injection_result.get("passed", False)
    screen_score = float(screen_result.get("screen_spoof_score", 1.0))
    screen_ok = not bool(screen_result.get("is_screen_replay", True))

    nid_matched = nid_match_result.get("is_same_person", False)
    nid_confidence = nid_match_result.get("confidence_score", 0.0)
    acct_matched = acct_match_result.get("is_same_person", False)
    acct_confidence = acct_match_result.get("confidence_score", 0.0)

    if SKIP_FACE_MATCH or match_source == "skipped":
        identity_ok = True
    elif match_result.get("fail_closed") or match_source == "fail_closed":
        identity_ok = False
    elif match_result.get("no_bank_reference"):
        identity_ok = False
    elif match_result.get("error") and not face_matched:
        identity_ok = nid_matched or acct_matched
    else:
        matched_src = match_result.get("matched_source")
        bank_overall = face_matched and matched_src in ("cbs_sign", "nid")
        stored_ok = False
        if match_source == "azure":
            stored_ok = face_matched and face_confidence >= AZURE_MATCH_THRESHOLD
        elif match_source == "external_api":
            stored_ok = face_matched and face_confidence >= IDENTITY_THRESHOLD
        elif match_source == "internal":
            stored_ok = bank_overall
        else:
            stored_ok = face_matched
        identity_ok = stored_ok or nid_matched or acct_matched

    temporal_score = float(temporal_result.get("liveness_score", 0.0))
    spatial_live_score = 1.0 - float(spatial_result.get("mean_spoof_score", 1.0))
    screen_live_score = max(0.0, 1.0 - screen_score)
    median_spoof = float(spatial_result.get("mean_spoof_score", 1.0))

    temporal_ok = temporal_live if REQUIRE_TEMPORAL else True
    spatial_ok = spatial_live if REQUIRE_SPATIAL else True
    screen_floor_ok = screen_ok

    w_sum = max(
        1e-6,
        LIVENESS_WEIGHT_SPATIAL + LIVENESS_WEIGHT_TEMPORAL + LIVENESS_WEIGHT_SCREEN,
    )
    composite_liveness = (
        LIVENESS_WEIGHT_SPATIAL * spatial_live_score
        + LIVENESS_WEIGHT_TEMPORAL * temporal_score
        + LIVENESS_WEIGHT_SCREEN * screen_live_score
    ) / w_sum
    composite_ok = composite_liveness >= COMPOSITE_LIVENESS_THRESHOLD

    # Hybrid step-up: borderline spatial + strong temporal/screen → challenge (never soft-pass)
    challenge: dict | None = None
    step_up = False
    if (
        STEP_UP_ENABLED
        and allow_step_up
        and REQUIRE_SPATIAL
        and not spatial_ok
        and screen_floor_ok
        and temporal_score >= STEP_UP_MIN_TEMPORAL
        and _spatial_is_borderline(spatial_result)
        and median_spoof < SPATIAL_BORDERLINE_HIGH
    ):
        step_up = True
        challenge = _pick_challenge()
        print(
            f"[STEP-UP] borderline spatial spoof={median_spoof:.3f} "
            f"pass_frac={spatial_result.get('pass_fraction')} "
            f"temporal={temporal_score:.3f} challenge={challenge}",
            flush=True,
        )

    live_ok = (
        False
        if step_up
        else (spatial_ok and screen_floor_ok and temporal_ok and composite_ok)
    )
    is_approved = (not step_up) and injection_passed and live_ok and identity_ok

    reason_codes: list[str] = []
    if is_approved:
        reason_codes.append("APPROVED")
    elif step_up:
        reason_codes.append("STEP_UP_REQUIRED")
    else:
        if not injection_passed:
            reason_codes.append("INJECTION_FAILED")
            if not injection_result.get("frame_count_valid", True):
                reason_codes.append("BAD_FRAME_COUNT")
            if not injection_result.get("consistent_resolution", True):
                reason_codes.append("INCONSISTENT_RESOLUTION")
            if not injection_result.get("timestamp_monotonic", True):
                reason_codes.append("TIMESTAMP_NOT_MONOTONIC")
        if REQUIRE_SPATIAL and not spatial_ok:
            reason_codes.append("SPATIAL_NOT_LIVE")
        if not screen_floor_ok:
            reason_codes.append("SCREEN_REPLAY")
        if REQUIRE_TEMPORAL and not temporal_live:
            reason_codes.append("TEMPORAL_NOT_LIVE")
        if spatial_ok and screen_floor_ok and temporal_ok and not composite_ok:
            reason_codes.append("COMPOSITE_LIVENESS_LOW")
        if not identity_ok:
            if match_result.get("fail_closed") or match_source == "fail_closed":
                reason_codes.append("FACE_MATCH_UNAVAILABLE")
            elif match_result.get("no_bank_reference"):
                reason_codes.append("NO_BANK_REFERENCE")
            else:
                reason_codes.append("IDENTITY_MISMATCH")

    print(
        f"[DECISION] injection={injection_passed} "
        f"spatial={spatial_live}(req={REQUIRE_SPATIAL}->{spatial_ok}) "
        f"temporal={temporal_live}(req={REQUIRE_TEMPORAL}->{temporal_ok}) "
        f"screen_ok={screen_floor_ok}({screen_score:.3f}) "
        f"composite={composite_liveness:.3f}(ok={composite_ok}) "
        f"step_up={step_up} identity={identity_ok} "
        f"APPROVED={is_approved} reasons={reason_codes}",
        flush=True,
    )
    logger.info(
        "DECISION session=%s approved=%s injection=%s spatial=%s temporal=%s "
        "screen=%.3f composite=%.3f step_up=%s identity=%s reasons=%s",
        session_id,
        is_approved,
        injection_passed,
        spatial_result.get("mean_spoof_score", 1.0),
        temporal_result.get("liveness_score", 0.0),
        screen_score,
        composite_liveness,
        step_up,
        identity_ok,
        reason_codes,
    )

    risk_components = [
        spatial_result.get("mean_spoof_score", 1.0),
        1.0 - temporal_score,
        screen_score,
        1.0 - composite_liveness,
        1.0 - face_confidence,
        0.0 if injection_passed else 0.5,
    ]
    composite_risk = sum(risk_components) / len(risk_components)

    result = VerificationResult(
        session_id=session_id,
        status=SessionStatus.COMPLETED,
        face_match=face_matched,
        face_match_confidence=face_confidence,
        nid_photo_match=(
            nid_matched
            if nid_photo_b64 or nid_match_result.get("template_count")
            else None
        ),
        nid_photo_match_confidence=(
            nid_confidence
            if nid_photo_b64 or nid_match_result.get("template_count")
            else None
        ),
        account_photo_match=(
            None
            if acct_match_result.get("error") in ("no_cbs", "no_templates")
            and not account_photo_b64
            else acct_matched
            if account_photo_b64
            or acct_match_result.get("template_count")
            or cbs_faces_detected
            else None
        ),
        account_photo_match_confidence=(
            acct_confidence
            if account_photo_b64
            or acct_match_result.get("template_count")
            or cbs_faces_detected
            else None
        ),
        nid_verified=nid_verified if nid_photo_b64 else None,
        customer_name=customer_name,
        face_match_provider=str(provider_used) if provider_used else active_code,
        cbs_faces_detected=cbs_faces_detected,
        liveness=live_ok,
        spatial_liveness_score=1.0 - spatial_result.get("mean_spoof_score", 1.0),
        temporal_liveness_score=temporal_result.get("liveness_score", 0.0),
        screen_spoof_score=screen_score,
        composite_risk_score=composite_risk,
        is_approved=is_approved,
        reason_codes=reason_codes,
        capture_trust=capture_trust,
        challenge=challenge,
        completed_at=datetime.now(timezone.utc),
    )

    result_dict = result.model_dump(mode="json")
    await log_verification(session_id, account_id, result_dict)

    # Persist training data from this mobile/test run
    try:
        await training_capture.capture_training_session(
            session_id=session_id,
            account_id=account_id,
            frames=frames,
            reference_image=reference_image,
            face_match_frame=match_meta.get("frame"),
            face_match_meta={
                "frame_index": match_meta.get("frame_index"),
                "blur_score": match_meta.get("blur_score"),
                "skip_face_match": SKIP_FACE_MATCH,
                "match_result": match_result,
            },
            result=result_dict,
            source=source,
            nid_photo_b64=nid_photo_b64,
            account_photo_b64=account_photo_b64,
        )
    except Exception:
        logger.exception("Training capture failed session=%s", session_id)
        print(f"[TRAINING] ERROR capturing session {session_id}", flush=True)

    print(
        f"########## VERIFY END session={session_id} approved={is_approved} ##########\n",
        flush=True,
    )
    return result
