"""Frame quality filters for video preprocessing."""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Phone camera + heavy JPEG/WebM compression often scores low Laplacian variance.
BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "12.0"))


def laplacian_variance(frame: np.ndarray) -> float:
    """Compute Laplacian variance as a blur metric. Higher = sharper."""
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_frame_sharp(frame: np.ndarray) -> bool:
    return laplacian_variance(frame) >= BLUR_THRESHOLD


def filter_sharp_frames(
    frames: list[np.ndarray],
    min_keep: int = 30,
    top_fraction: float = 0.5,
) -> list[np.ndarray]:
    """Keep only sharp frames for spatial/temporal models.

    Strategy:
      1. Prefer frames above BLUR_THRESHOLD.
      2. Also keep the sharpest top_fraction of the clip (relative quality).
      3. Intersection if large enough; else absolute-threshold set; else top-N.
    Never feed the blurriest frames into UAD / 3D-CNN when better ones exist.
    """
    if not frames:
        return []

    scored = [(laplacian_variance(f), i) for i, f in enumerate(frames)]
    scored.sort(key=lambda x: x[0], reverse=True)

    abs_idx = {i for v, i in scored if v >= BLUR_THRESHOLD}
    top_n = max(min_keep, int(len(scored) * top_fraction))
    rel_idx = {i for _, i in scored[:top_n]}

    chosen_idx = abs_idx & rel_idx
    if len(chosen_idx) < min_keep:
        chosen_idx = abs_idx if len(abs_idx) >= min_keep else rel_idx

    if not chosen_idx:
        keep_n = min(len(scored), max(min_keep, 30))
        chosen_idx = {i for _, i in scored[:keep_n]}
        best = scored[0][0] if scored else 0.0
        logger.warning(
            "No frames above blur threshold %.1f (best=%.2f); using top %d frames",
            BLUR_THRESHOLD,
            best,
            len(chosen_idx),
        )

    ordered = [frames[i] for i in sorted(chosen_idx)]
    worst_kept = min(laplacian_variance(frames[i]) for i in chosen_idx)
    print(
        f"[QUALITY] frames={len(frames)} kept={len(ordered)} "
        f"blur_thr={BLUR_THRESHOLD} best={scored[0][0]:.1f} worst_kept={worst_kept:.1f}",
        flush=True,
    )
    return ordered


def select_spatial_samples(frames: list[np.ndarray], max_samples: int = 10) -> list[np.ndarray]:
    """Evenly sample across the sharp set, biased to highest quality."""
    sharp = filter_sharp_frames(frames, min_keep=max(max_samples, 20))
    if not sharp:
        return []
    if len(sharp) <= max_samples:
        return sharp
    ranked = sorted(sharp, key=laplacian_variance, reverse=True)
    pool = ranked[: max(max_samples * 2, len(ranked) // 2)]
    step = max(1, len(pool) // max_samples)
    return pool[::step][:max_samples]


def get_best_quality_frame(frames: list[np.ndarray]) -> np.ndarray | None:
    """Return the sharpest frame."""
    if not frames:
        return None
    return max(frames, key=laplacian_variance)


def _mean_brightness(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 2:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def assess_capture_quality(
    frames: list[np.ndarray],
    *,
    min_blur: float | None = None,
    min_brightness: float = 40.0,
    max_brightness: float = 220.0,
    sample_n: int = 12,
) -> dict:
    """Strict capture-quality gate before treating a PAD fail as definitive.

    Poor blur/lighting should surface as CAPTURE_QUALITY (retry), not as a
    reason to bypass spatial checks.
    """
    min_blur = BLUR_THRESHOLD if min_blur is None else float(min_blur)
    if not frames:
        return {
            "adequate": False,
            "median_blur": 0.0,
            "mean_brightness": 0.0,
            "frames_checked": 0,
            "reasons": ["NO_FRAMES"],
        }

    step = max(1, len(frames) // sample_n)
    sampled = frames[::step][:sample_n]
    blurs = [laplacian_variance(f) for f in sampled]
    brights = [_mean_brightness(f) for f in sampled]
    median_blur = float(np.median(blurs))
    mean_brightness = float(np.mean(brights))

    reasons: list[str] = []
    if median_blur < min_blur:
        reasons.append("BLUR_TOO_LOW")
    if mean_brightness < min_brightness:
        reasons.append("TOO_DARK")
    if mean_brightness > max_brightness:
        reasons.append("TOO_BRIGHT")

    adequate = not reasons
    print(
        f"[QUALITY-GATE] adequate={adequate} median_blur={median_blur:.1f} "
        f"(min={min_blur}) brightness={mean_brightness:.1f} "
        f"[{min_brightness},{max_brightness}] reasons={reasons}",
        flush=True,
    )
    return {
        "adequate": adequate,
        "median_blur": median_blur,
        "mean_brightness": mean_brightness,
        "frames_checked": len(sampled),
        "reasons": reasons,
    }
