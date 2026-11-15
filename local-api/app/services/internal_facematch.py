"""Internal face match: MTCNN multi-face templates + ArcFace cosine / pgvector.

Pipeline:
1. prepare_account_templates — extract faces from CBS signature card + NID, store in faceprints
2. match_live_to_account — embed live frame, cosine vs all templates (any-of for joint accounts)
3. match_live_to_reference_b64 — ad-hoc single-image internal match (NID-only path)
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import cv2
import numpy as np

from app.db import pgvector
from app.services import gpu_client

logger = logging.getLogger(__name__)

INTERNAL_FACE_MATCH_THRESHOLD = float(os.getenv("INTERNAL_FACE_MATCH_THRESHOLD", "0.45"))


def _decode_b64_image(b64: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-8))


def _sim_to_confidence(sim: float) -> float:
    if sim < 0:
        return max(0.0, min(1.0, (sim + 1.0) / 2.0))
    return min(1.0, sim / 0.6) if sim < 0.6 else min(1.0, 0.85 + (sim - 0.6) * 0.375)


async def prepare_account_templates(
    account_id: str,
    *,
    nid_photo_b64: Optional[str] = None,
    account_photo_b64: Optional[str] = None,
) -> dict:
    """Extract and store face templates for CBS sign + NID.

    Returns:
      cbs_faces_detected, nid_faces_detected, errors[]
    """
    result = {
        "cbs_faces_detected": 0,
        "nid_faces_detected": 0,
        "errors": [],
        "account_id": account_id,
    }

    if account_photo_b64:
        img = _decode_b64_image(account_photo_b64)
        if img is None:
            result["errors"].append("Invalid CBS signature card image")
        else:
            try:
                faces = await gpu_client.extract_face_templates(
                    img, document_mode=True, min_prob=0.85, include_crop_jpeg=True
                )
                rows = []
                for f in faces:
                    rows.append({
                        "embedding": f["embedding"],
                        "reference_image_b64": f.get("crop_jpeg_b64"),
                        "face_index": int(f.get("face_index", len(rows))),
                        "bbox": f.get("bbox"),
                        "provider_code": "internal",
                    })
                await pgvector.replace_account_faceprints(account_id, "cbs_sign", rows)
                result["cbs_faces_detected"] = len(rows)
                if not rows:
                    result["errors"].append(
                        "Customer photo not found on signature card"
                    )
                print(
                    f"[INTERNAL FM] cbs_sign account={account_id} faces={len(rows)}",
                    flush=True,
                )
            except Exception as e:
                logger.exception("CBS template extract failed")
                result["errors"].append(f"CBS template extract failed: {e}")
    else:
        result["errors"].append("Customer photo not found for this account")
        await pgvector.replace_account_faceprints(account_id, "cbs_sign", [])

    if nid_photo_b64:
        img = _decode_b64_image(nid_photo_b64)
        if img is None:
            result["errors"].append("Invalid NID photo")
        else:
            try:
                faces = await gpu_client.extract_face_templates(
                    img, document_mode=True, min_prob=0.85, include_crop_jpeg=True
                )
                rows = []
                for f in faces:
                    rows.append({
                        "embedding": f["embedding"],
                        "reference_image_b64": f.get("crop_jpeg_b64") or nid_photo_b64.split(",")[-1],
                        "face_index": int(f.get("face_index", len(rows))),
                        "bbox": f.get("bbox"),
                        "provider_code": "internal",
                    })
                # If MTCNN missed, still store full-image embedding
                if not rows:
                    emb = await gpu_client.get_embedding(img)
                    if emb:
                        rows.append({
                            "embedding": emb,
                            "reference_image_b64": nid_photo_b64.split(",")[-1],
                            "face_index": 0,
                            "bbox": None,
                            "provider_code": "internal",
                        })
                await pgvector.replace_account_faceprints(account_id, "nid", rows)
                result["nid_faces_detected"] = len(rows)
                print(
                    f"[INTERNAL FM] nid account={account_id} faces={len(rows)}",
                    flush=True,
                )
            except Exception as e:
                logger.exception("NID template extract failed")
                result["errors"].append(f"NID template extract failed: {e}")

    return result


async def match_live_to_account(
    live_image: np.ndarray,
    account_id: str,
    *,
    sources: Optional[list[str]] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Match live face against stored templates for account (any-of).

    Default sources exclude client_photo (the selfie just captured).
    """
    thr = INTERNAL_FACE_MATCH_THRESHOLD if threshold is None else threshold
    src = sources if sources is not None else ["cbs_sign", "nid"]
    live_emb = await gpu_client.get_embedding(live_image)
    if not live_emb:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "no_embedding",
            "source": "internal",
            "provider_used": "internal",
        }

    hit = await pgvector.match_embedding_to_account(
        account_id, live_emb, sources=src
    )
    if hit.get("template_count", 0) == 0:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "no_templates",
            "source": "internal",
            "provider_used": "internal",
            "template_count": 0,
        }

    sim = float(hit.get("cosine_similarity") or 0.0)
    matched = sim >= thr
    conf = _sim_to_confidence(sim)
    return {
        "is_same_person": matched,
        "confidence_score": conf if matched else max(0.0, conf * 0.5),
        "cosine_similarity": sim,
        "matched_source": hit.get("matched_source"),
        "matched_face_index": hit.get("matched_face_index"),
        "template_count": hit.get("template_count"),
        "threshold": thr,
        "source": "internal",
        "provider_used": "internal",
    }


async def match_live_to_reference_b64(
    live_image: np.ndarray,
    reference_image_b64: str,
    *,
    threshold: Optional[float] = None,
    document_mode: bool = True,
) -> dict:
    """Ad-hoc internal match: live vs a single reference image (may contain multiple faces)."""
    thr = INTERNAL_FACE_MATCH_THRESHOLD if threshold is None else threshold
    ref = _decode_b64_image(reference_image_b64)
    if ref is None:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "bad_reference",
            "source": "internal",
            "provider_used": "internal",
        }

    live_emb = await gpu_client.get_embedding(live_image)
    if not live_emb:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "no_embedding",
            "source": "internal",
            "provider_used": "internal",
        }

    try:
        faces = await gpu_client.extract_face_templates(
            ref, document_mode=document_mode, include_crop_jpeg=False
        )
    except Exception as e:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": f"template_extract_failed: {e}",
            "source": "internal",
            "provider_used": "internal",
        }

    embeddings = [f["embedding"] for f in faces if f.get("embedding")]
    if not embeddings:
        ref_emb = await gpu_client.get_embedding(ref)
        if ref_emb:
            embeddings = [ref_emb]

    if not embeddings:
        return {
            "is_same_person": False,
            "confidence_score": 0.0,
            "error": "no_face_in_reference",
            "source": "internal",
            "provider_used": "internal",
        }

    best_sim = -1.0
    best_idx = 0
    for i, emb in enumerate(embeddings):
        sim = _cosine(live_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    matched = best_sim >= thr
    conf = _sim_to_confidence(best_sim)
    return {
        "is_same_person": matched,
        "confidence_score": conf if matched else max(0.0, conf * 0.5),
        "cosine_similarity": best_sim,
        "matched_face_index": best_idx,
        "face_count_on_card": len(embeddings),
        "threshold": thr,
        "source": "internal",
        "provider_used": "internal",
    }


async def match_live_split_by_source(
    live_image: np.ndarray,
    account_id: str,
    *,
    threshold: Optional[float] = None,
) -> dict:
    """Run separate CBS and NID template matches for result fields."""
    thr = INTERNAL_FACE_MATCH_THRESHOLD if threshold is None else threshold
    cbs = await match_live_to_account(
        live_image, account_id, sources=["cbs_sign"], threshold=thr
    )
    nid = await match_live_to_account(
        live_image, account_id, sources=["nid"], threshold=thr
    )
    # Overall: any of cbs/nid/other stored
    overall = await match_live_to_account(live_image, account_id, threshold=thr)
    return {
        "overall": overall,
        "cbs": cbs,
        "nid": nid,
        "cbs_faces": cbs.get("template_count") or 0,
        "nid_faces": nid.get("template_count") or 0,
    }
