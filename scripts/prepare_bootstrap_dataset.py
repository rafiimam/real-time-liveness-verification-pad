#!/usr/bin/env python3
"""Build a bootstrap FAS + temporal dataset when academic dumps are unavailable.

Creates face-crop-like images and short video clips under:
    <out>/fas_merged/{real,spoof}/
    <out>/temporal_merged/{real,fake}/

Real class: natural photos / mild augmentations.
Spoof class: print-like (low contrast, moiré, border) and screen-replay artifacts.

This is for pipeline bring-up and checkpoint wiring — replace with CelebA-Spoof /
CASIA / Replay-Attack for production accuracy.

Usage:
    python scripts/prepare_bootstrap_dataset.py --output /data --n-images 2000 --n-videos 40
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np


def _face_like(rng: random.Random, size: int = 224) -> np.ndarray:
    """Synthesize a crude face-like blob with lighting variation."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    skin = np.array([rng.randint(90, 200), rng.randint(70, 160), rng.randint(50, 120)], dtype=np.int16)
    cx, cy = size // 2 + rng.randint(-10, 10), size // 2 + rng.randint(-15, 10)
    axes = (size // 3 + rng.randint(-8, 8), size // 2 + rng.randint(-10, 10))
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, skin.tolist(), -1)
    # eyes
    eye_y = cy - size // 10
    for ex in (cx - size // 8, cx + size // 8):
        cv2.circle(img, (ex, eye_y), size // 28, (20, 20, 20), -1)
        cv2.circle(img, (ex + 2, eye_y - 1), size // 55, (220, 220, 220), -1)
    # mouth
    cv2.ellipse(img, (cx, cy + size // 7), (size // 10, size // 28), 0, 0, 180, (40, 30, 80), 2)
    # soft lighting
    yy, xx = np.mgrid[0:size, 0:size]
    light = 0.85 + 0.25 * np.sin(xx / size * math.pi + rng.random())
    img = np.clip(img.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)
    noise = rng.randint(3, 12)
    noise_arr = np.random.default_rng(rng.randint(0, 10_000_000)).integers(-noise, noise + 1, size=img.shape)
    img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)
    return img


def _spoof_print(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = cv2.GaussianBlur(img, (5, 5), 0)
    out = cv2.convertScaleAbs(out, alpha=0.75, beta=15)
    # paper border
    h, w = out.shape[:2]
    border = max(4, min(h, w) // 20)
    out = cv2.copyMakeBorder(out, border, border, border, border, cv2.BORDER_CONSTANT, value=(230, 230, 230))
    out = cv2.resize(out, (w, h))
    # slight moiré / texture
    yy, xx = np.mgrid[0:h, 0:w]
    moire = (12 * np.sin(xx / 3.0 + rng.random() * 5)).astype(np.int16)
    out = np.clip(out.astype(np.int16) + moire[..., None], 0, 255).astype(np.uint8)
    return out


def _spoof_screen(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # RGB subpixel stripes
    stripe = np.zeros_like(out)
    stripe[:, ::3, 0] = 18
    stripe[:, 1::3, 1] = 18
    stripe[:, 2::3, 2] = 18
    out = np.clip(out.astype(np.int16) + stripe, 0, 255).astype(np.uint8)
    # glare
    overlay = out.copy()
    cv2.circle(overlay, (rng.randint(w // 4, 3 * w // 4), rng.randint(h // 5, h // 2)), w // 6, (255, 255, 255), -1)
    out = cv2.addWeighted(out, 0.82, overlay, 0.18, 0)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def write_images(out_root: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    real_dir = out_root / "fas_merged" / "real"
    spoof_dir = out_root / "fas_merged" / "spoof"
    real_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        face = _face_like(rng)
        cv2.imwrite(str(real_dir / f"real_{i:05d}.jpg"), face)
        spoof = _spoof_print(face, rng) if i % 2 == 0 else _spoof_screen(face, rng)
        cv2.imwrite(str(spoof_dir / f"spoof_{i:05d}.jpg"), spoof)
    print(f"Wrote {n} real + {n} spoof images under {out_root / 'fas_merged'}")


def write_videos(out_root: Path, n: int, seed: int, frames: int = 45, fps: int = 8) -> None:
    rng = random.Random(seed + 99)
    real_dir = out_root / "temporal_merged" / "real"
    fake_dir = out_root / "temporal_merged" / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    size = 112
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for i in range(n):
        base = _face_like(rng, size=size)
        # real: small natural motion / lighting drift
        path = real_dir / f"real_{i:04d}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
        for t in range(frames):
            frame = base.copy()
            dx = int(3 * math.sin(t / 5.0))
            M = np.float32([[1, 0, dx], [0, 1, int(2 * math.cos(t / 7.0))]])
            frame = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REFLECT)
            scale = 0.95 + 0.08 * math.sin(t / 9.0)
            frame = cv2.convertScaleAbs(frame, alpha=scale, beta=0)
            writer.write(frame)
        writer.release()

        # fake: static / replay with screen stripes + freeze frames
        path = fake_dir / f"fake_{i:04d}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
        spoof_base = _spoof_screen(base, rng)
        for t in range(frames):
            # mostly frozen with occasional jump (replay artifact)
            if t % 12 < 10:
                frame = spoof_base
            else:
                frame = cv2.convertScaleAbs(spoof_base, alpha=1.0, beta=rng.randint(-8, 8))
            writer.write(frame)
        writer.release()

    print(f"Wrote {n} real + {n} fake videos under {out_root / 'temporal_merged'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap FAS/temporal training data")
    parser.add_argument("--output", required=True, help="Output root (creates fas_merged + temporal_merged)")
    parser.add_argument("--n-images", type=int, default=2000)
    parser.add_argument("--n-videos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_images(out, args.n_images, args.seed)
    write_videos(out, args.n_videos, args.seed)
    print("Bootstrap dataset ready. For production, replace with CelebA-Spoof / CASIA / Replay-Attack.")


if __name__ == "__main__":
    main()
