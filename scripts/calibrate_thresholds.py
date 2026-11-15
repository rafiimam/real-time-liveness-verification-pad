#!/usr/bin/env python3
"""Calibrate spatial / temporal liveness thresholds on a held-out set.

Spatial (images):
    --fas-dir with real/ and spoof/  → spoof_score distributions
    Recommend SPATIAL_LIVENESS_THRESHOLD such that spoof FAR ≈ target.

Temporal (videos):
    --temporal-dir with real/ and fake/ → liveness_score distributions
    Recommend TEMPORAL_LIVENESS_THRESHOLD.

Can score via GPU HTTP API or local model load.

Examples:
    # Via running GPU server
    python scripts/calibrate_thresholds.py \\
        --fas-dir /data/fas_merged --gpu-url http://127.0.0.1:8000 \\
        --target-far 0.01 --max-samples 500

    # Local weights (run from gpu-server venv)
    python scripts/calibrate_thresholds.py \\
        --fas-dir /data/fas_merged --local-weights \\
        --swin-weights gpu-server/weights/swin_uad_best.pth
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.suffix.lower() in exts)


def encode_jpeg_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf).decode()


def percentile_threshold(spoof_scores: list[float], target_far: float) -> float:
    """For spoof scores (higher = more spoof): real pass if score < thr.

    Choose thr so that fraction of spoof samples with score < thr ≈ target_far.
    """
    if not spoof_scores:
        return 0.5
    arr = np.sort(np.asarray(spoof_scores, dtype=np.float64))
    idx = max(0, int(np.floor(target_far * len(arr))) - 1)
    # Allow at most target_far of spoofs below threshold
    return float(arr[idx]) if target_far > 0 else float(arr[0])


def temporal_threshold(fake_liveness: list[float], target_far: float) -> float:
    """Live if liveness > thr. Choose thr so fraction of fakes above thr ≈ FAR."""
    if not fake_liveness:
        return 0.9
    arr = np.sort(np.asarray(fake_liveness, dtype=np.float64))
    # thr such that P(fake > thr) ≈ FAR → thr ≈ (1-FAR) quantile
    q = 1.0 - target_far
    return float(np.quantile(arr, q))


def score_spatial_http(paths: list[Path], gpu_url: str) -> list[float]:
    import httpx

    scores = []
    with httpx.Client(timeout=60.0) as client:
        for p in paths:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            r = client.post(
                f"{gpu_url.rstrip('/')}/spoof/frame",
                json={"image_base64": encode_jpeg_b64(bgr)},
            )
            r.raise_for_status()
            scores.append(float(r.json()["spoof_score"]))
    return scores


def score_temporal_http(paths: list[Path], gpu_url: str, max_frames: int = 60) -> list[float]:
    import httpx

    scores = []
    with httpx.Client(timeout=180.0) as client:
        for p in paths:
            cap = cv2.VideoCapture(str(p))
            frames_b64 = []
            while cap.isOpened() and len(frames_b64) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames_b64.append(encode_jpeg_b64(frame))
            cap.release()
            if len(frames_b64) < 8:
                continue
            r = client.post(
                f"{gpu_url.rstrip('/')}/spoof/temporal",
                json={"frames_base64": frames_b64},
            )
            r.raise_for_status()
            scores.append(float(r.json()["liveness_score"]))
    return scores


def score_spatial_local(paths: list[Path], weights: str) -> list[float]:
    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "gpu-server"))
    os.environ.setdefault("SWIN_UAD_WEIGHTS", weights)

    import torch
    from app.models.swin_backbone import get_swin_model
    from app.routers.spoof import face_crop_to_swin_tensor
    from app.config import DEVICE

    model = get_swin_model()
    scores = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        tensor = face_crop_to_swin_tensor(bgr)
        if tensor is None:
            # full-frame fallback for synthetic face-like images without MTCNN hit
            rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
            tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            s = model.get_spoof_score(tensor.unsqueeze(0).to(DEVICE))
        scores.append(float(s[0]))
    return scores


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate liveness thresholds")
    parser.add_argument("--fas-dir", help="Image dataset root with real/ and spoof/")
    parser.add_argument("--temporal-dir", help="Video dataset root with real/ and fake/")
    parser.add_argument("--gpu-url", default="", help="GPU API base URL")
    parser.add_argument("--local-weights", action="store_true")
    parser.add_argument("--swin-weights", default="")
    parser.add_argument("--target-far", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    report: dict = {"target_far": args.target_far, "recommendations": {}}

    if args.fas_dir:
        root = Path(args.fas_dir)
        real = list_files(root / "real", IMAGE_EXTS)[: args.max_samples]
        spoof = list_files(root / "spoof", IMAGE_EXTS)[: args.max_samples]
        print(f"Spatial: scoring {len(real)} real + {len(spoof)} spoof")

        if args.local_weights or not args.gpu_url:
            weights = args.swin_weights or str(
                Path(__file__).resolve().parent.parent / "gpu-server" / "weights" / "swin_uad_best.pth"
            )
            real_scores = score_spatial_local(real, weights)
            spoof_scores = score_spatial_local(spoof, weights)
        else:
            real_scores = score_spatial_http(real, args.gpu_url)
            spoof_scores = score_spatial_http(spoof, args.gpu_url)

        thr = percentile_threshold(spoof_scores, args.target_far)
        # Ensure most reals still pass: thr should be above real p90 when possible
        if real_scores:
            real_p90 = float(np.quantile(real_scores, 0.90))
            # Use max of FAR-based thr and a small margin over real distribution midpoint
            thr = float(max(thr, min(0.5, real_p90 + 0.05)))
        # If spoofs are saturated at 1.0, FAR quantile is 1.0 — keep a usable mid threshold
        if thr >= 0.99 and real_scores and float(np.quantile(real_scores, 0.90)) < 0.5:
            thr = 0.5

        report["spatial"] = {
            "real": summarize("real", real_scores),
            "spoof": summarize("spoof", spoof_scores),
            "recommended_SPATIAL_LIVENESS_THRESHOLD": thr,
            "note": "Approve when mean_spoof_score < threshold (lower spoof = more live)",
        }
        report["recommendations"]["SPATIAL_LIVENESS_THRESHOLD"] = thr
        print(json.dumps(report["spatial"], indent=2))

    if args.temporal_dir:
        if not args.gpu_url:
            print("Temporal calibration requires --gpu-url (HTTP /spoof/temporal)")
        else:
            root = Path(args.temporal_dir)
            real = list_files(root / "real", VIDEO_EXTS)[: args.max_samples]
            fake = list_files(root / "fake", VIDEO_EXTS)[: args.max_samples]
            print(f"Temporal: scoring {len(real)} real + {len(fake)} fake videos")
            real_liv = score_temporal_http(real, args.gpu_url)
            fake_liv = score_temporal_http(fake, args.gpu_url)
            thr = temporal_threshold(fake_liv, args.target_far)
            if real_liv:
                real_p10 = float(np.quantile(real_liv, 0.10))
                # Keep thr below most real scores but prefer a high bar when separation is clear
                if real_p10 > 0.85 and (not fake_liv or float(np.quantile(fake_liv, 0.90)) < 0.5):
                    thr = 0.9
                else:
                    thr = float(min(thr, max(0.5, real_p10 - 0.02)))
            report["temporal"] = {
                "real": summarize("real", real_liv),
                "fake": summarize("fake", fake_liv),
                "recommended_TEMPORAL_LIVENESS_THRESHOLD": thr,
                "note": "Approve when liveness_score > threshold",
            }
            report["recommendations"]["TEMPORAL_LIVENESS_THRESHOLD"] = thr
            print(json.dumps(report["temporal"], indent=2))

    print("\n=== Recommended env ===")
    for k, v in report["recommendations"].items():
        print(f"{k}={v:.4f}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
