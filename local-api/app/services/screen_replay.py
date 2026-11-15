"""Screen / replay detection for phone captures.

Phone WebM compression makes CFA moiré / refresh bands fire on BOTH live and
screen-filmed video. Those cues are logged but not used alone for the decision.

Decision relies on:
  - RGB subpixel striping (stronger when filming an LCD up close)
  - Soft detail / low Laplacian energy (distant PC playback recapture)
  - Temporal difference row-bands (refresh / rolling alias)
  - Non-rigid face motion residual (stronger for live faces — but also
    fires on *played* face video, so live-protect only applies when sharp)
  - Bezel / glare when visible
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCREEN_SPOOF_THRESHOLD = float(os.getenv("SCREEN_SPOOF_THRESHOLD", "0.46"))


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _resize(frame: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _center_crop(img: np.ndarray, frac: float = 0.60) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * frac), int(w * frac)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0 : y0 + ch, x0 : x0 + cw]


def _glare_score(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    mask = (v > 238) & (s < 35)
    ratio = float(mask.mean())
    return float(np.clip((ratio - 0.01) / 0.06, 0.0, 1.0))


def _bezel_rectangle_score(gray: np.ndarray) -> float:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 60, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    h, w = gray.shape[:2]
    area_img = float(h * w)
    best = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.12 * area_img or area > 0.88 * area_img:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        rect = cv2.minAreaRect(approx)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        aspect = max(rw, rh) / min(rw, rh)
        if aspect > 2.6:
            continue
        fill = area / area_img
        best = max(best, float(np.clip((fill - 0.18) / 0.50, 0.0, 1.0)))
    return best


def _rgb_stripe_score(bgr: np.ndarray) -> float:
    """LCD RGB subpixel column periodicity — strongest cue for filming a monitor."""
    img = _center_crop(bgr).astype(np.float32)
    if img.shape[1] < 24:
        return 0.0
    channels = [img[:, :, c] for c in range(3)]
    best = 0.0
    for ch in channels:
        hp = ch[:, 1:] - ch[:, :-1]
        col_energy = np.mean(np.abs(hp), axis=0)
        for p in (2, 3):
            a, b = col_energy[:-p], col_energy[p:]
            if a.std() < 1e-6:
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            best = max(best, max(0.0, corr))
    return float(np.clip((best - 0.45) / 0.45, 0.0, 1.0))


def _rgb_stripe_multiscale(bgr: np.ndarray) -> float:
    """Stripe score across a few crop/scales (helps mid-distance monitors)."""
    scores = [_rgb_stripe_score(bgr)]
    for scale, frac in ((400, 0.70), (520, 0.75)):
        img = _center_crop(_resize(bgr, scale), frac).astype(np.float32)
        if img.shape[1] < 24:
            continue
        best = 0.0
        for c in range(3):
            ch = img[:, :, c]
            hp = ch[:, 1:] - ch[:, :-1]
            col_energy = np.mean(np.abs(hp), axis=0)
            for p in (2, 3, 4):
                a, b = col_energy[:-p], col_energy[p:]
                if a.std() < 1e-6:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                best = max(best, max(0.0, corr))
        scores.append(float(np.clip((best - 0.45) / 0.45, 0.0, 1.0)))
    return float(max(scores)) if scores else 0.0


def _soft_recapture_raw(frames: list[np.ndarray]) -> float:
    """Mean Laplacian variance of face-ish center crop (lower = softer)."""
    if not frames:
        return 200.0
    step = max(1, len(frames) // 10)
    vals: list[float] = []
    for f in frames[::step][:10]:
        g = _center_crop(_to_gray(_resize(f, 360)), 0.6).astype(np.float32)
        vals.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
    return float(np.mean(vals)) if vals else 200.0


def _soft_recapture_score(raw: float) -> float:
    """Distant PC playback is often soft; sharp live phone faces are not."""
    return float(np.clip((140.0 - raw) / 100.0, 0.0, 1.0))


def _temporal_band_peak_score(frames: list[np.ndarray]) -> float:
    """Peak row-FFT of inter-frame diffs — refresh / rolling alias cue."""
    if len(frames) < 8:
        return 0.0
    step = max(1, len(frames) // 16)
    sampled = [_to_gray(_resize(f, 320)) for f in frames[::step][:16]]
    peaks: list[float] = []
    for i in range(1, len(sampled)):
        d = np.abs(sampled[i].astype(np.float32) - sampled[i - 1].astype(np.float32))
        row = d.mean(axis=1)
        row = row - row.mean()
        if row.std() < 0.5:
            continue
        spec = np.abs(np.fft.rfft(row * np.hanning(len(row))))
        spec[0] = 0
        peaks.append(float(spec.max() / (spec.mean() + 1e-6)))
    if not peaks:
        return 0.0
    return float(np.clip((float(np.max(peaks)) - 14.0) / 20.0, 0.0, 1.0))


def _moire_peak_score(gray: np.ndarray) -> float:
    """Diagnostic only — unreliable on compressed phone video."""
    g = _center_crop(gray).astype(np.float32)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    spec = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))
    cy, cx = spec.shape[0] // 2, spec.shape[1] // 2
    spec[cy - 5 : cy + 6, cx - 5 : cx + 6] = 0
    flat = spec.ravel()
    if flat.size < 100:
        return 0.0
    top = np.partition(flat, -10)[-10:]
    med = float(np.median(flat) + 1e-6)
    peak_ratio = float(top.mean() / med)
    return float(np.clip((peak_ratio - 18.0) / 28.0, 0.0, 1.0))


def _refresh_band_score(frames: list[np.ndarray]) -> float:
    """Diagnostic only — unreliable on compressed phone video."""
    if len(frames) < 10:
        return 0.0
    step = max(1, len(frames) // 20)
    sampled = frames[::step][:20]
    band_hits = 0
    for f in sampled:
        g = _to_gray(_resize(f, 240)).astype(np.float32)
        row_means = g.mean(axis=1)
        row_means = row_means - row_means.mean()
        if row_means.std() < 1.5:
            continue
        spec = np.abs(np.fft.rfft(row_means * np.hanning(len(row_means))))
        if spec.size < 5:
            continue
        spec[0] = 0
        peak = float(spec.max() / (spec.mean() + 1e-6))
        if peak > 12.0:
            band_hits += 1
    return float(np.clip(band_hits / max(1, len(sampled) * 0.45), 0.0, 1.0))


def _nonrigid_live_score(frames: list[np.ndarray]) -> float:
    """Higher = more non-rigid face motion (live *or* played face video)."""
    if len(frames) < 8:
        return 0.5

    step = max(1, len(frames) // 16)
    idxs = list(range(0, len(frames), step))[:16]
    residual_ratios = []

    for i in range(len(idxs) - 1):
        a = _center_crop(_to_gray(_resize(frames[idxs[i]], 360)), 0.7)
        b = _center_crop(_to_gray(_resize(frames[idxs[i + 1]], 360)), 0.7)
        pts = cv2.goodFeaturesToTrack(
            a, maxCorners=120, qualityLevel=0.01, minDistance=6, blockSize=7
        )
        if pts is None or len(pts) < 20:
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is None:
            continue
        status = status.reshape(-1)
        good_old = pts[status == 1].reshape(-1, 2)
        good_new = nxt[status == 1].reshape(-1, 2)
        if len(good_old) < 20:
            continue
        motion = float(np.linalg.norm(good_new - good_old, axis=1).mean())
        if motion < 0.35:
            continue
        M, _inliers = cv2.estimateAffinePartial2D(
            good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=2.5
        )
        if M is None:
            continue
        ones = np.ones((len(good_old), 1), dtype=np.float32)
        homo = np.hstack([good_old.astype(np.float32), ones])
        pred = (M @ homo.T).T
        resid = np.linalg.norm(pred - good_new.astype(np.float32), axis=1)
        rel_resid = float(np.median(resid)) / (motion + 1e-3)
        residual_ratios.append(rel_resid)

    if not residual_ratios:
        return 0.4
    mean_rel = float(np.mean(residual_ratios))
    return float(np.clip((mean_rel - 0.05) / 0.40, 0.0, 1.0))


def analyze_screen_replay(frames: list[np.ndarray], max_samples: int = 12) -> dict:
    if not frames:
        return {
            "screen_spoof_score": 1.0,
            "is_screen_replay": True,
            "components": {},
            "frames_analyzed": 0,
        }

    step = max(1, len(frames) // max_samples)
    sampled = frames[::step][:max_samples]

    glare, bezel, moire, stripe = [], [], [], []
    for f in sampled:
        small = _resize(f, 320)
        g = _to_gray(small)
        glare.append(_glare_score(small))
        bezel.append(_bezel_rectangle_score(g))
        moire.append(_moire_peak_score(g))
        stripe.append(_rgb_stripe_multiscale(small))

    soft_raw = _soft_recapture_raw(frames)
    soft = _soft_recapture_score(soft_raw)
    tb = _temporal_band_peak_score(frames)
    nonrigid = _nonrigid_live_score(frames)
    components = {
        "rgb_stripe": float(np.mean(stripe)),
        "nonrigid_live": nonrigid,
        "soft_recapture": soft,
        "temporal_band_peak": tb,
        "glare": float(np.mean(glare)),
        "bezel": float(np.mean(bezel)),
        # diagnostic only
        "soft_laplacian_var": soft_raw,
        "moire": float(np.mean(moire)),
        "refresh_band": _refresh_band_score(frames),
    }

    # Fusion: soft catches distant PC playback; stripe+planar catches close screens.
    # Do not trust nonrigid alone — played face videos also look non-rigid.
    score = (
        0.45 * components["rgb_stripe"]
        + 0.30 * (1.0 - components["nonrigid_live"])
        + 0.20 * components["soft_recapture"]
        + 0.05 * components["bezel"]
    )
    score = float(np.clip(score, 0.0, 1.0))

    # Distant PC playback: soft + glare/temporal bands
    if soft >= 0.45 and components["glare"] >= 0.30 and tb >= 0.40:
        score = max(score, 0.55)
    if soft >= 0.60 and (components["glare"] >= 0.25 or tb >= 0.55):
        score = max(score, 0.58)
    if soft >= 0.50 and tb >= 0.80:
        score = max(score, 0.62)

    # Close screen: strong striping + planar motion
    if components["rgb_stripe"] >= 0.75 and nonrigid < 0.18:
        score = max(score, 0.55)
    if components["bezel"] >= 0.35 and nonrigid < 0.40:
        score = max(score, 0.60)

    # Live protect — only when sharp (not soft recapture). Played video can have
    # high nonrigid from content; soft+bands still flag those as screen.
    # Very sharp phone frames (soft≈0) often sit just under the old 0.18 nonrigid
    # floor and were false-flagged SCREEN_REPLAY despite strong spatial/temporal.
    if soft < 0.12:
        keep_screen = (
            components["rgb_stripe"] >= 0.70
            and nonrigid < 0.15
            and components["bezel"] >= 0.20
        )
        if not keep_screen:
            score = min(score, 0.40)
    if soft < 0.30 and nonrigid >= 0.15:
        keep_screen = tb >= 0.75 and components["rgb_stripe"] >= 0.65 and nonrigid < 0.30
        if not keep_screen:
            score = min(score, 0.40)
    if soft < 0.22 and nonrigid >= 0.45:
        score = min(score, 0.36)

    score = float(np.clip(score, 0.0, 1.0))
    is_screen = score >= SCREEN_SPOOF_THRESHOLD
    logger.info(
        "Screen replay analysis score=%.3f threshold=%.3f components=%s",
        score,
        SCREEN_SPOOF_THRESHOLD,
        {k: round(v, 3) if isinstance(v, float) else v for k, v in components.items()},
    )
    return {
        "screen_spoof_score": score,
        "is_screen_replay": is_screen,
        "components": components,
        "frames_analyzed": len(sampled),
    }
