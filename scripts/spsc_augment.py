"""Simulated Physical Spoofing Clues (SPSC) + phone-domain real augmentations.

Spoof path: moiré / resample, RGB stripe, color cast, JPEG recompress, blur+sharpen
Real path: mild ISP noise, exposure jitter, phone JPEG — teach UAD to ignore sensor noise
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _to_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _from_np(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def jpeg_recompress(img: Image.Image, quality: int | None = None) -> Image.Image:
    q = quality if quality is not None else random.randint(35, 85)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def add_gaussian_noise(img: Image.Image, sigma: float | None = None) -> Image.Image:
    arr = _to_np(img).astype(np.float32)
    s = sigma if sigma is not None else random.uniform(2.0, 12.0)
    noise = np.random.normal(0.0, s, arr.shape).astype(np.float32)
    return _from_np(arr + noise)


def exposure_jitter(img: Image.Image, amount: float | None = None) -> Image.Image:
    a = amount if amount is not None else random.uniform(0.75, 1.35)
    return ImageEnhance.Brightness(img).enhance(a)


def color_cast(img: Image.Image) -> Image.Image:
    arr = _to_np(img).astype(np.float32)
    shifts = np.array(
        [
            random.uniform(-18, 18),
            random.uniform(-18, 18),
            random.uniform(-18, 18),
        ],
        dtype=np.float32,
    )
    return _from_np(arr + shifts)


def rgb_stripe(img: Image.Image, strength: float | None = None) -> Image.Image:
    """Vertical chroma stripes resembling display scan / capture artifacts."""
    arr = _to_np(img).astype(np.float32)
    h, w, _ = arr.shape
    s = strength if strength is not None else random.uniform(6.0, 22.0)
    period = random.randint(2, 5)
    xs = np.arange(w, dtype=np.float32)
    wave = np.sin(2 * np.pi * xs / period) * s
    arr[:, :, 0] += wave
    arr[:, :, 1] -= wave * 0.5
    arr[:, :, 2] += wave * 0.35
    return _from_np(arr)


def moire_resample(img: Image.Image) -> Image.Image:
    """Down/up sample at mismatched scales to induce moiré-like interference."""
    w, h = img.size
    scale = random.uniform(0.35, 0.65)
    small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
    # Second scale mismatch
    mid_scale = random.uniform(0.7, 1.15)
    mid = small.resize(
        (max(8, int(w * mid_scale)), max(8, int(h * mid_scale))),
        Image.NEAREST if random.random() < 0.5 else Image.BILINEAR,
    )
    back = mid.resize((w, h), Image.BILINEAR)
    # Blend with original lightly
    a = random.uniform(0.45, 0.85)
    return Image.blend(img, back, a)


def blur_sharpen(img: Image.Image) -> Image.Image:
    radius = random.uniform(0.6, 1.8)
    soft = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return soft.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))


def apply_spsc_spoof(img: Image.Image) -> Image.Image:
    """Stack 1–3 spoof-oriented transforms."""
    ops = [moire_resample, rgb_stripe, color_cast, blur_sharpen, jpeg_recompress]
    random.shuffle(ops)
    out = img
    for op in ops[: random.randint(1, 3)]:
        out = op(out)
    if random.random() < 0.5:
        out = jpeg_recompress(out, quality=random.randint(30, 70))
    return out


def apply_phone_real(img: Image.Image) -> Image.Image:
    """Mild ISP / compression noise typical of live phone captures."""
    out = img
    if random.random() < 0.7:
        out = exposure_jitter(out)
    if random.random() < 0.6:
        out = add_gaussian_noise(out, sigma=random.uniform(1.5, 8.0))
    if random.random() < 0.8:
        out = jpeg_recompress(out, quality=random.randint(55, 92))
    if random.random() < 0.25:
        out = ImageEnhance.Contrast(out).enhance(random.uniform(0.9, 1.15))
    return out


def augment_for_label(img: Image.Image, label: int) -> Image.Image:
    """label: 0=real, 1=spoof."""
    if int(label) == 1:
        if random.random() < 0.85:
            return apply_spsc_spoof(img)
        return img
    if random.random() < 0.75:
        return apply_phone_real(img)
    return img
