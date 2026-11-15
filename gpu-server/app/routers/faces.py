"""Multi-face template extraction for signature cards / documents.

POST /faces/templates
  - MTCNN detect (document_mode lowers min face size)
  - ArcFace 512-D embedding per face
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.models.arcface_head import get_embedder
from app.models.mtcnn_detector import get_detector, get_document_detector

router = APIRouter()


class TemplatesRequest(BaseModel):
    image_base64: str
    document_mode: bool = True
    min_prob: float = Field(default=0.85, ge=0.0, le=1.0)
    include_crop_jpeg: bool = True


class FaceTemplate(BaseModel):
    face_index: int
    bbox: list[float]
    confidence: float
    embedding: list[float]
    dimension: int
    crop_jpeg_b64: str | None = None


class TemplatesResponse(BaseModel):
    faces: list[FaceTemplate]
    face_count: int
    document_mode: bool


def _crop_padded(image: np.ndarray, box: list[float], pad: float = 0.25) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2].copy()


@router.post("/templates", response_model=TemplatesResponse)
async def extract_face_templates(req: TemplatesRequest):
    img_bytes = base64.b64decode(req.image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return TemplatesResponse(faces=[], face_count=0, document_mode=req.document_mode)

    detector = get_document_detector() if req.document_mode else get_detector()
    # Slightly lower min_prob in document mode (small/scanned faces)
    min_prob = req.min_prob if not req.document_mode else min(req.min_prob, 0.85)
    detections = detector.detect_and_align(image, min_prob=min_prob)

    embedder = get_embedder()
    faces_out: list[FaceTemplate] = []

    for i, det in enumerate(detections):
        box = det["box"]
        crop_bgr = _crop_padded(image, box)
        emb = embedder.get_embedding_from_frame(crop_bgr)
        if emb is None and det.get("crop") is not None:
            emb = embedder.get_embedding_from_crop(det["crop"])
            if emb is not None and float(np.linalg.norm(emb)) < 1e-6:
                emb = None
        if emb is None:
            continue

        crop_b64 = None
        if req.include_crop_jpeg:
            _, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            crop_b64 = base64.b64encode(buf).decode()

        faces_out.append(
            FaceTemplate(
                face_index=i,
                bbox=[float(x) for x in box[:4]],
                confidence=float(det["confidence"]),
                embedding=emb.tolist(),
                dimension=len(emb),
                crop_jpeg_b64=crop_b64,
            )
        )

    # Fallback: InsightFace multi-detect if MTCNN found nothing
    if not faces_out:
        insight_faces = embedder.app.get(image)
        for i, f in enumerate(insight_faces or []):
            if float(getattr(f, "det_score", 0)) < min_prob:
                continue
            bbox = f.bbox.tolist() if hasattr(f.bbox, "tolist") else list(f.bbox)
            crop_b64 = None
            if req.include_crop_jpeg:
                crop_bgr = _crop_padded(image, bbox)
                _, buf = cv2.imencode(".jpg", crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                crop_b64 = base64.b64encode(buf).decode()
            faces_out.append(
                FaceTemplate(
                    face_index=i,
                    bbox=[float(x) for x in bbox[:4]],
                    confidence=float(f.det_score),
                    embedding=f.normed_embedding.tolist(),
                    dimension=len(f.normed_embedding),
                    crop_jpeg_b64=crop_b64,
                )
            )

    return TemplatesResponse(
        faces=faces_out,
        face_count=len(faces_out),
        document_mode=req.document_mode,
    )
