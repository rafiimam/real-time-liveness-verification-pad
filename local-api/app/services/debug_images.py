"""Show debug images in console (ASCII) and optional popup viewer (Spyder-like)."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SHOW_DEBUG_IMAGES = os.getenv("SHOW_DEBUG_IMAGES", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
ASCII_WIDTH = int(os.getenv("ASCII_IMAGE_WIDTH", "64"))


def _to_ascii(image: np.ndarray, width: int = ASCII_WIDTH) -> str:
    chars = np.asarray(list(" .:-=+*#%@"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    new_w = max(20, width)
    new_h = max(8, int(h * (new_w / w) * 0.45))
    small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    idx = (small.astype(np.float32) / 255.0 * (len(chars) - 1)).astype(np.int32)
    lines = ["".join(chars[row].tolist()) for row in idx]
    return "\n".join(lines)


def print_image_console(image: np.ndarray | None, title: str = "image") -> None:
    """Print ASCII preview to console (always) and optionally open a GUI window."""
    if image is None or getattr(image, "size", 0) == 0:
        print(f"[IMAGE] {title}: <empty>", flush=True)
        return

    h, w = image.shape[:2]
    print(f"\n[IMAGE] ===== {title} ({w}x{h}) =====", flush=True)
    try:
        print(_to_ascii(image), flush=True)
    except Exception as e:
        print(f"[IMAGE] ASCII preview failed: {e}", flush=True)
    print(f"[IMAGE] ===== end {title} =====\n", flush=True)

    if not SHOW_DEBUG_IMAGES:
        return

    def _open():
        try:
            # Prefer system viewer (works without OpenCV GUI / Spyder-like popup)
            from PIL import Image

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            # Downscale huge phone photos for display
            max_side = 900
            if max(pil.size) > max_side:
                scale = max_side / max(pil.size)
                pil = pil.resize((int(pil.size[0] * scale), int(pil.size[1] * scale)))
            pil.show(title=title)
            logger.info("Opened image viewer for %s", title)
        except Exception as e1:
            try:
                fd, path = tempfile.mkstemp(prefix="liveness_", suffix=".jpg")
                os.close(fd)
                cv2.imwrite(path, image)
                if os.name == "nt":
                    os.startfile(path)  # type: ignore[attr-defined]
                else:
                    import subprocess

                    subprocess.Popen(["xdg-open", path])
                logger.info("Opened OS viewer for %s -> %s", title, path)
            except Exception as e2:
                logger.warning("Could not open image viewer (%s / %s)", e1, e2)

    threading.Thread(target=_open, daemon=True).start()
