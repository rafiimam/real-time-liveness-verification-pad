"""HTTP client for GPU inference server."""

import base64
import os
from typing import Optional

import cv2
import httpx
import numpy as np

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.getenv("GPU_TIMEOUT", "120"))
TEMPORAL_TIMEOUT = float(os.getenv("GPU_TEMPORAL_TIMEOUT", "180"))


async def detect_faces(image: np.ndarray) -> list[dict]:
    """Call GPU /detect endpoint."""
    _, buf = cv2.imencode(".jpg", image)
    b64 = base64.b64encode(buf).decode()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{GPU_SERVER_URL}/detect", json={"image_base64": b64})
        resp.raise_for_status()
        return resp.json()["faces"]


async def get_embedding(image: np.ndarray) -> Optional[list[float]]:
    """Call GPU /embedding endpoint."""
    _, buf = cv2.imencode(".jpg", image)
    b64 = base64.b64encode(buf).decode()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{GPU_SERVER_URL}/embedding", json={"image_base64": b64})
        resp.raise_for_status()
        data = resp.json()
        if data["dimension"] == 0:
            return None
        return data["embedding"]


async def extract_face_templates(
    image: np.ndarray,
    *,
    document_mode: bool = True,
    min_prob: float = 0.85,
    include_crop_jpeg: bool = True,
) -> list[dict]:
    """Call GPU /faces/templates — MTCNN multi-face + ArcFace embeddings.

    Returns list of {face_index, bbox, confidence, embedding, crop_jpeg_b64?}.
    """
    _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buf).decode()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(
            f"{GPU_SERVER_URL}/faces/templates",
            json={
                "image_base64": b64,
                "document_mode": document_mode,
                "min_prob": min_prob,
                "include_crop_jpeg": include_crop_jpeg,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return list(data.get("faces") or [])


async def get_spoof_score(image: np.ndarray) -> float:
    """Call GPU /spoof/frame endpoint."""
    _, buf = cv2.imencode(".jpg", image)
    b64 = base64.b64encode(buf).decode()

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{GPU_SERVER_URL}/spoof/frame", json={"image_base64": b64})
        resp.raise_for_status()
        return resp.json()["spoof_score"]


async def get_temporal_liveness(frames: list[np.ndarray]) -> dict:
    """Call GPU /spoof/temporal endpoint with frame batch."""
    frames_b64 = []
    for frame in frames:
        _, buf = cv2.imencode(".jpg", frame)
        frames_b64.append(base64.b64encode(buf).decode())

    async with httpx.AsyncClient(timeout=TEMPORAL_TIMEOUT) as client:
        resp = await client.post(
            f"{GPU_SERVER_URL}/spoof/temporal",
            json={"frames_base64": frames_b64},
        )
        resp.raise_for_status()
        return resp.json()


async def process_full_stream(video_bytes: bytes) -> dict:
    """Upload full video to GPU /process-stream endpoint."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{GPU_SERVER_URL}/process-stream",
            files={"video": ("stream.mp4", video_bytes, "video/mp4")},
        )
        resp.raise_for_status()
        return resp.json()


async def finetune_uad(
    samples: list[tuple[np.ndarray, int]],
    epochs: int = 2,
    lr: float = 1e-4,
    promote: bool = False,
) -> dict:
    """POST labeled frames to GPU /spoof/finetune (0=real, 1=spoof)."""
    payload = []
    for image, label in samples:
        _, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        payload.append(
            {
                "image_base64": base64.b64encode(buf).decode(),
                "label": int(label),
            }
        )
    async with httpx.AsyncClient(timeout=600.0) as client:
        resp = await client.post(
            f"{GPU_SERVER_URL}/spoof/finetune",
            json={
                "samples": payload,
                "epochs": epochs,
                "lr": lr,
                "promote": bool(promote),
            },
        )
        resp.raise_for_status()
        return resp.json()
