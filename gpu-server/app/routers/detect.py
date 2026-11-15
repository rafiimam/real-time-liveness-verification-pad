"""Face detection endpoint."""

import base64

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.mtcnn_detector import get_detector

router = APIRouter()


class DetectRequest(BaseModel):
    image_base64: str


class DetectedFace(BaseModel):
    box: list[float]
    confidence: float
    landmarks: list[list[float]]
    crop_base64: str


class DetectResponse(BaseModel):
    faces: list[DetectedFace]


@router.post("", response_model=DetectResponse)
async def detect_face(req: DetectRequest):
    img_bytes = base64.b64decode(req.image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    detector = get_detector()
    faces = detector.detect_and_align(image)

    results = []
    for f in faces:
        crop_np = ((f["crop"].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", crop_np[:, :, ::-1])
        crop_b64 = base64.b64encode(buf).decode()
        results.append(DetectedFace(
            box=f["box"],
            confidence=f["confidence"],
            landmarks=f["landmarks"],
            crop_base64=crop_b64,
        ))

    return DetectResponse(faces=results)
