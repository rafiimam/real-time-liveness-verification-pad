"""Face match clients.

Providers (DB-selectable via face_match_providers):
1 polygon — Bank face-match API
2 azure — Azure Face Verify
3 internal — MTCNN + ArcFace cosine / pgvector

Active provider is used first; on outage only, fall back 3 → 1 → 2.
"""

import base64
import logging
import os

import cv2
import httpx
import numpy as np

logger = logging.getLogger(__name__)

FACEMATCH_URL = os.getenv(
    "FACEMATCH_URL").strip()
TIMEOUT = float(os.getenv("FACEMATCH_TIMEOUT", "60"))

# Azure Face — override via env in production
AZURE_FACE_ENDPOINT = os.getenv(
    "AZURE_FACE_ENDPOINT").rstrip("/") + "/"
AZURE_FACE_KEY = os.getenv(
    "AZURE_FACE_SUBSCRIPTION_KEY")
AZURE_MATCH_THRESHOLD = float(os.getenv("AZURE_FACE_MATCH_THRESHOLD", "0.7"))


async def match_faces(
    live_image: np.ndarray,
    reference_image_b64: str,
) -> dict:
    """Compare a live capture against a stored reference image (bank API)."""
    _, buf = cv2.imencode(".jpg", live_image)
    live_b64 = base64.b64encode(buf).decode()

    payload = {
        "image_1": live_b64,
        "image_2": reference_image_b64,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                FACEMATCH_URL,
                json=payload,
                headers={"accept": "application/json", "Content-Type": "application/json"},
            )
            # Bank middleware returns 502 when Ekyc vendor is down — keep body for logs
            if resp.status_code >= 400:
                print(f"[FACE MATCH] bank API HTTP {resp.status_code}: {resp.text[:400]}", flush=True)
            resp.raise_for_status()
            data = resp.json()
            print(
                f"[FACE MATCH] bank API ok keys={list(data.keys()) if isinstance(data, dict) else type(data)}",
                flush=True,
            )
    except Exception as e:
        logger.warning("External face-match API failed: %s", e)
        print(f"[FACE MATCH] bank API error: {e}", flush=True)
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": f"facematch_api_error: {type(e).__name__}: {e}",
            "source": "external_api",
        }

    if isinstance(data, dict) and data.get("success"):
        inner = data.get("data") or {}
        return {
            "is_same_person": bool(inner.get("is_same_person", inner.get("isSamePerson", False))),
            "confidence_score": float(
                inner.get("confidence_score", inner.get("confidenceScore", 0.0)) or 0.0
            ),
            "source": "external_api",
        }
    if isinstance(data, dict) and ("is_same_person" in data or "isSamePerson" in data):
        return {
            "is_same_person": bool(data.get("is_same_person", data.get("isSamePerson", False))),
            "confidence_score": float(
                data.get("confidence_score", data.get("confidenceScore", 0.0)) or 0.0
            ),
            "source": "external_api",
        }
    return {
        "is_same_person": False,
        "confidence_score": 0.0,
        "error": data.get("message", "face match unsuccessful")
        if isinstance(data, dict)
        else "unexpected_response",
        "source": "external_api",
        "raw": data if isinstance(data, dict) else str(data)[:500],
    }


async def _azure_detect_face_id(client: httpx.AsyncClient, image_bytes: bytes) -> str | None:
    """Detect the largest face and return its faceId."""
    url = (
        f"{AZURE_FACE_ENDPOINT}face/v1.0/detect"
        "?returnFaceId=true"
        "&returnFaceLandmarks=false"
        "&recognitionModel=recognition_04"
        "&detectionModel=detection_03"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_FACE_KEY,
        "Content-Type": "application/octet-stream",
    }
    resp = await client.post(url, content=image_bytes, headers=headers)
    resp.raise_for_status()
    faces = resp.json()
    if not faces:
        return None
    # Prefer largest face by bounding box area
    best = max(
        faces,
        key=lambda f: f["faceRectangle"]["width"] * f["faceRectangle"]["height"],
    )
    return best.get("faceId")


async def match_faces_via_azure(
    live_image: np.ndarray,
    reference_image_b64: str,
    threshold: float | None = None,
) -> dict:
    """Backup identity check using Azure Face Verify."""
    threshold = AZURE_MATCH_THRESHOLD if threshold is None else threshold

    if not AZURE_FACE_KEY:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "azure_key_missing",
            "source": "azure",
        }

    _, live_buf = cv2.imencode(".jpg", live_image)
    live_bytes = live_buf.tobytes()
    try:
        ref_bytes = base64.b64decode(reference_image_b64)
    except Exception as e:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": f"bad_reference_b64: {e}",
            "source": "azure",
        }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            live_id = await _azure_detect_face_id(client, live_bytes)
            ref_id = await _azure_detect_face_id(client, ref_bytes)
            if not live_id or not ref_id:
                return {
                    "is_same_person": False,
                    "confidence_score": 0.0,
                    "error": "azure_no_face_detected",
                    "source": "azure",
                }

            verify_url = f"{AZURE_FACE_ENDPOINT}face/v1.0/verify"
            headers = {
                "Ocp-Apim-Subscription-Key": AZURE_FACE_KEY,
                "Content-Type": "application/json",
            }
            resp = await client.post(
                verify_url,
                headers=headers,
                json={"faceId1": live_id, "faceId2": ref_id},
            )
            resp.raise_for_status()
            data = resp.json()

        confidence = float(data.get("confidence", 0.0))
        is_identical = bool(data.get("isIdentical", False))
        # Also enforce our configured threshold
        matched = is_identical and confidence >= threshold
        return {
            "is_same_person": matched,
            "confidence_score": confidence,
            "azure_is_identical": is_identical,
            "source": "azure",
        }
    except Exception as e:
        logger.warning("Azure face-match failed: %s", e)
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": f"azure_error: {type(e).__name__}: {e}",
            "source": "azure",
        }


async def match_faces_via_gpu(
    live_image: np.ndarray,
    reference_image_b64: str,
    threshold: float = 0.35,
) -> dict:
    """Last-resort identity check using GPU ArcFace embeddings + cosine similarity."""
    from app.services import gpu_client

    img_bytes = base64.b64decode(reference_image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    ref_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if ref_image is None:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "bad_reference",
            "source": "gpu_fallback",
        }

    live_emb = await gpu_client.get_embedding(live_image)
    ref_emb = await gpu_client.get_embedding(ref_image)
    if not live_emb or not ref_emb:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "no_embedding",
            "source": "gpu_fallback",
        }

    a = np.asarray(live_emb, dtype=np.float32)
    b = np.asarray(ref_emb, dtype=np.float32)
    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    confidence = max(0.0, min(1.0, (sim + 1.0) / 2.0)) if sim < 0 else min(1.0, sim / 0.6)
    return {
        "is_same_person": sim >= threshold,
        "confidence_score": confidence if sim >= threshold else max(0.0, confidence * 0.5),
        "cosine_similarity": sim,
        "source": "gpu_fallback",
    }


FACE_MATCH_FAIL_CLOSED = os.getenv("FACE_MATCH_FAIL_CLOSED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

INTERNAL_FACE_MATCH_THRESHOLD = float(os.getenv("INTERNAL_FACE_MATCH_THRESHOLD", "0.45"))

# Provider codes
PROVIDER_POLYGON = "polygon"
PROVIDER_AZURE = "azure"
PROVIDER_INTERNAL = "internal"


async def _call_provider(
    code: str,
    live_image: np.ndarray,
    reference_image_b64: str | None,
    *,
    account_id: str | None = None,
) -> dict:
    """Dispatch to a single provider. Returns result with provider_used."""
    if code == PROVIDER_POLYGON:
        print(f"[FACE MATCH] trying Polygon/bank API {FACEMATCH_URL}", flush=True)
        if not reference_image_b64:
            return {
                "is_same_person": False,
                "confidence_score": 0.0,
                "error": "no_reference",
                "source": "external_api",
                "provider_used": PROVIDER_POLYGON,
            }
        r = await match_faces(live_image, reference_image_b64)
        r["provider_used"] = PROVIDER_POLYGON
        return r

    if code == PROVIDER_AZURE:
        print("[FACE MATCH] trying Azure Face", flush=True)
        if not reference_image_b64:
            return {
                "is_same_person": False,
                "confidence_score": 0.0,
                "error": "no_reference",
                "source": "azure",
                "provider_used": PROVIDER_AZURE,
            }
        r = await match_faces_via_azure(live_image, reference_image_b64)
        r["provider_used"] = PROVIDER_AZURE
        return r

    # internal
    print("[FACE MATCH] trying Internal MTCNN+ArcFace", flush=True)
    from app.services import internal_facematch

    if account_id:
        r = await internal_facematch.match_live_to_account(
            live_image, account_id, threshold=INTERNAL_FACE_MATCH_THRESHOLD
        )
        if r.get("error") != "no_templates":
            r["provider_used"] = PROVIDER_INTERNAL
            return r
        # Never fall back to client selfie when account_id is known —
        # identity must come from CBS signature card and/or NID photo.
        print(
            f"[FACE MATCH] no CBS/NID templates for account={account_id} — identity fail",
            flush=True,
        )
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "source": "internal",
            "provider_used": PROVIDER_INTERNAL,
            "message": "no_reference_templates",
            "no_bank_reference": True,
        }

    if reference_image_b64:
        r = await internal_facematch.match_live_to_reference_b64(
            live_image,
            reference_image_b64,
            threshold=INTERNAL_FACE_MATCH_THRESHOLD,
            document_mode=True,
        )
        r["provider_used"] = PROVIDER_INTERNAL
        return r
    return {
        "is_same_person": False,
        "confidence_score": 0.0,
        "source": "internal",
        "provider_used": PROVIDER_INTERNAL,
        "message": "no_reference_templates",
        "no_bank_reference": True,
    }


async def match_faces_routed(
    live_image: np.ndarray,
    reference_image_b64: str | None = None,
    *,
    account_id: str | None = None,
) -> dict:
    """Use DB-active provider; on outage only, fall back 3→1→2.

    A false match (is_same_person=False without error) does NOT hop providers.
    """
    from app.db.pgvector import get_active_face_match_provider, list_enabled_providers_fallback_order

    active = await get_active_face_match_provider()
    active_code = (active.get("code") or PROVIDER_INTERNAL).strip().lower()
    print(
        f"[FACE MATCH] active_provider id={active.get('id')} code={active_code}",
        flush=True,
    )

    ordered = await list_enabled_providers_fallback_order()
    # Active first, then remaining in fallback order
    codes: list[str] = []
    if active_code:
        codes.append(active_code)
    for p in ordered:
        c = p.get("code")
        if c and c not in codes:
            codes.append(c)

    errors: list[str] = []
    last: dict = {}
    for i, code in enumerate(codes):
        result = await _call_provider(
            code, live_image, reference_image_b64, account_id=account_id
        )
        last = result
        if not result.get("error"):
            if errors:
                result["external_error"] = " | ".join(errors)
                result["fallback_used"] = True
            print(f"[FACE MATCH] provider={code} success: {result}", flush=True)
            return result
        # Outage — try next
        err = str(result.get("error"))
        errors.append(f"{code}:{err}")
        print(f"[FACE MATCH] provider={code} outage: {err}", flush=True)
        if i + 1 < len(codes):
            print(f"[FACE MATCH] falling back to {codes[i + 1]}", flush=True)

    chain = " | ".join(errors)
    if FACE_MATCH_FAIL_CLOSED:
        print(f"[FACE MATCH] FAIL-CLOSED all providers failed: {chain}", flush=True)
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "all_face_match_providers_failed",
            "external_error": chain,
            "source": "fail_closed",
            "fail_closed": True,
            "provider_used": last.get("provider_used"),
        }
    last["external_error"] = chain
    return last


async def match_faces_with_fallback(
    live_image: np.ndarray,
    reference_image_b64: str,
    *,
    account_id: str | None = None,
) -> dict:
    """Compat wrapper — routes via DB-active provider."""
    return await match_faces_routed(
        live_image, reference_image_b64, account_id=account_id
    )
