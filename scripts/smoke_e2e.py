"""Smoke-test GPU endpoints and local-api e2e flow."""

import base64
import json
from pathlib import Path

import cv2
import httpx
import numpy as np

GPU = "http://localhost:8000"
LOCAL = "http://127.0.0.1:8001"
SAMPLE_PATH = "sample_face.jpg"


def load_sample_image() -> np.ndarray:
    if not Path(SAMPLE_PATH).exists():
        raise FileNotFoundError(f"Missing {SAMPLE_PATH}")
    img = cv2.imread(SAMPLE_PATH)
    if img is None:
        raise RuntimeError("Failed to decode sample image")
    # Upscale small portraits so detectors get enough pixels
    if img.shape[0] < 256 or img.shape[1] < 256:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img


def to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()


def make_video(frames: list[np.ndarray], path: str, fps: int = 30) -> str:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def main():
    print("=== Loading sample face ===")
    face = load_sample_image()
    print(f"  image shape: {face.shape}")

    b64 = to_b64(face)
    client = httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0))

    print("\n=== GPU /detect ===")
    r = client.post(f"{GPU}/detect", json={"image_base64": b64})
    print(f"  status={r.status_code}")
    detect = r.json()
    print(f"  faces={len(detect.get('faces', []))}")
    if detect.get("faces"):
        print(f"  first confidence={detect['faces'][0]['confidence']:.3f}")

    print("\n=== GPU /embedding ===")
    r = client.post(f"{GPU}/embedding", json={"image_base64": b64})
    print(f"  status={r.status_code}")
    emb = r.json()
    print(f"  dim={emb.get('dimension')}")

    print("\n=== GPU /spoof/frame ===")
    r = client.post(f"{GPU}/spoof/frame", json={"image_base64": b64})
    print(f"  status={r.status_code} body={r.text[:300]}")

    print("\n=== GPU /spoof/temporal ===")
    frames_b64 = [b64] * 30
    r = client.post(f"{GPU}/spoof/temporal", json={"frames_base64": frames_b64})
    print(f"  status={r.status_code} body={r.text[:300]}")

    print("\n=== LOCAL /account/enroll ===")
    r = client.post(f"{LOCAL}/account/enroll", json={
        "account_id": "TEST_ACC_001",
        "image_base64": b64,
        "embedding": emb.get("embedding") or [0.0] * 512,
    })
    print(f"  status={r.status_code} body={r.text}")

    print("\n=== LOCAL /session/create ===")
    r = client.post(f"{LOCAL}/session/create", json={
        "account_id": "TEST_ACC_001",
        "transaction_id": "TXN_SMOKE_1",
        "transaction_amount": 100000.0,
    })
    print(f"  status={r.status_code} body={r.text}")
    session = r.json()
    sid = session["session_id"]

    print("\n=== Build 10s video & upload ===")
    # slight jitter so frames aren't identical
    video_frames = []
    for i in range(300):
        f = face.copy()
        shift = (i % 5) - 2
        M = np.float32([[1, 0, shift], [0, 1, 0]])
        f = cv2.warpAffine(f, M, (f.shape[1], f.shape[0]))
        video_frames.append(f)
    video_path = "smoke_test.mp4"
    make_video(video_frames, video_path)
    print(f"  wrote {video_path}")

    with open(video_path, "rb") as f:
        r = client.post(
            f"{LOCAL}/session/{sid}/stream",
            files={"video": ("smoke_test.mp4", f, "video/mp4")},
            timeout=300.0,
        )
    print(f"  status={r.status_code}")
    try:
        print(f"  result={json.dumps(r.json(), indent=2, default=str)}")
    except Exception:
        print(f"  raw={r.text[:1000]}")

    print("\n=== LOCAL /session/{id}/result ===")
    r = client.get(f"{LOCAL}/session/{sid}/result")
    print(f"  status={r.status_code} body={r.text}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
