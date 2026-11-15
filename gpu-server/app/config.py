import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "weights")
FACE_SIZE = 112
SWIN_INPUT_SIZE = 224
TEMPORAL_CHUNK_FRAMES = 30

# Prefer finetuned checkpoint; fall back to best / env override
_SWIN_DEFAULT = os.path.join(WEIGHTS_DIR, "swin_uad_finetuned.pth")
_SWIN_FALLBACK = os.path.join(WEIGHTS_DIR, "swin_uad_best.pth")
SWIN_UAD_WEIGHTS = os.getenv("SWIN_UAD_WEIGHTS", "")
if not SWIN_UAD_WEIGHTS:
    SWIN_UAD_WEIGHTS = _SWIN_DEFAULT if os.path.isfile(_SWIN_DEFAULT) else _SWIN_FALLBACK

TEMPORAL_WEIGHTS = os.getenv(
    "TEMPORAL_WEIGHTS",
    os.path.join(WEIGHTS_DIR, "temporal_3dcnn_best.pth"),
)

SPATIAL_SPOOF_THRESHOLD = float(os.getenv("SPATIAL_SPOOF_THRESHOLD", "0.5"))
TEMPORAL_LIVENESS_THRESHOLD = float(os.getenv("TEMPORAL_LIVENESS_THRESHOLD", "0.9"))

# When true, prefer MTCNN crops (matches CelebA-Spoof-style FAS data).
# Set false for bootstrap/full-frame trained checkpoints.
USE_FACE_CROP = os.getenv("USE_FACE_CROP", "true").strip().lower() in (
    "1", "true", "yes", "on",
)
