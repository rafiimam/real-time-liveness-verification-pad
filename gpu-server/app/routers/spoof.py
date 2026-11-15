"""Spoof detection endpoints: per-frame (spatial) and temporal.

Spatial scoring prefers MTCNN-aligned face crops resized to 224x224 so
inference matches FAS training data. Falls back to full-frame resize when
no face is found (or USE_FACE_CROP=false for bootstrap checkpoints).
"""

import base64
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import (
    DEVICE,
    SPATIAL_SPOOF_THRESHOLD,
    SWIN_INPUT_SIZE,
    TEMPORAL_LIVENESS_THRESHOLD,
    USE_FACE_CROP,
)
from app.models.mtcnn_detector import get_detector
from app.models.swin_backbone import get_swin_model

logger = logging.getLogger(__name__)
router = APIRouter()


def _decode_bgr(image_b64: str) -> np.ndarray | None:
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def full_frame_to_swin_tensor(bgr: np.ndarray) -> torch.Tensor:
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (SWIN_INPUT_SIZE, SWIN_INPUT_SIZE))
    tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    return (tensor - 0.5) / 0.5


def face_crop_to_swin_tensor(bgr: np.ndarray) -> torch.Tensor | None:
    """MTCNN-align best face, resize to 224x224 Swin input in [-1, 1]."""
    detector = get_detector()
    crop = detector.get_best_face(bgr)  # (3, 112, 112) in roughly [-1, 1]
    if crop is None:
        return None
    crop = crop.unsqueeze(0)
    crop = F.interpolate(
        crop, size=(SWIN_INPUT_SIZE, SWIN_INPUT_SIZE), mode="bilinear", align_corners=False
    )
    return crop.squeeze(0)


def prepare_swin_input(bgr: np.ndarray) -> tuple[torch.Tensor, bool]:
    """Return (tensor, face_detected)."""
    if USE_FACE_CROP:
        crop = face_crop_to_swin_tensor(bgr)
        if crop is not None:
            return crop, True
        logger.warning("No face for spatial score — falling back to full frame")
    return full_frame_to_swin_tensor(bgr), False


def face_crop_to_temporal_rgb(bgr: np.ndarray) -> np.ndarray | None:
    detector = get_detector()
    crop = detector.get_best_face(bgr)
    if crop is None:
        return None
    arr = ((crop.detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


def prepare_temporal_rgb(bgr: np.ndarray) -> np.ndarray:
    if USE_FACE_CROP:
        crop = face_crop_to_temporal_rgb(bgr)
        if crop is not None:
            return crop
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (112, 112))


class SpoofFrameRequest(BaseModel):
    image_base64: str


class SpoofFrameResponse(BaseModel):
    spoof_score: float
    is_live: bool
    face_detected: bool = True


@router.post("/frame", response_model=SpoofFrameResponse)
async def spoof_frame(req: SpoofFrameRequest):
    model = get_swin_model()
    bgr = _decode_bgr(req.image_base64)
    if bgr is None:
        return SpoofFrameResponse(spoof_score=1.0, is_live=False, face_detected=False)

    tensor, face_detected = prepare_swin_input(bgr)
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = model.get_spoof_score(tensor)

    spoof_val = float(score[0])
    return SpoofFrameResponse(
        spoof_score=spoof_val,
        is_live=spoof_val < SPATIAL_SPOOF_THRESHOLD,
        face_detected=face_detected,
    )


class SpoofTemporalRequest(BaseModel):
    frames_base64: list[str]


class SpoofTemporalResponse(BaseModel):
    liveness_score: float
    is_live: bool
    chunk_scores: list[float]
    faces_used: int = 0


@router.post("/temporal", response_model=SpoofTemporalResponse)
async def spoof_temporal(req: SpoofTemporalRequest):
    """Temporal liveness via 3D CNN on face crops (or full frames)."""
    from app.models.temporal_3dcnn import get_temporal_model, CHUNK_FRAMES

    model = get_temporal_model()
    frames = []
    faces_used = 0
    for b64 in req.frames_base64:
        bgr = _decode_bgr(b64)
        if bgr is None:
            continue
        if USE_FACE_CROP:
            crop = face_crop_to_temporal_rgb(bgr)
            if crop is not None:
                frames.append(crop)
                faces_used += 1
                continue
        frames.append(prepare_temporal_rgb(bgr))

    if not frames:
        return SpoofTemporalResponse(
            liveness_score=0.0,
            is_live=False,
            chunk_scores=[],
            faces_used=0,
        )

    frames_np = np.array(frames, dtype=np.float32) / 255.0

    chunk_scores = []
    for i in range(0, len(frames_np), CHUNK_FRAMES):
        chunk = frames_np[i : i + CHUNK_FRAMES]
        if len(chunk) < CHUNK_FRAMES:
            pad = np.zeros((CHUNK_FRAMES - len(chunk), *chunk.shape[1:]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=0)

        tensor = torch.from_numpy(chunk).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            score = model(tensor)
        chunk_scores.append(float(score[0]))

    avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 1.0
    liveness = 1.0 - avg_score

    return SpoofTemporalResponse(
        liveness_score=liveness,
        is_live=liveness > TEMPORAL_LIVENESS_THRESHOLD,
        chunk_scores=chunk_scores,
        faces_used=faces_used,
    )


class FinetuneSample(BaseModel):
    image_base64: str
    label: int  # 0=real, 1=spoof


class FinetuneRequest(BaseModel):
    samples: list[FinetuneSample]
    epochs: int = 2
    lr: float = 1e-4
    promote: bool = False


class FinetuneResponse(BaseModel):
    status: str
    n_real: int = 0
    n_spoof: int = 0
    epochs: int = 0
    mean_loss: float | None = None
    weights: str | None = None
    device: str | None = None
    promoted: bool = False
    error: str | None = None


def _run_uad_finetune(
    samples: list[FinetuneSample],
    epochs: int,
    lr: float,
    promote: bool = False,
) -> FinetuneResponse:
    """Fine-tune UAD head only; save checkpoint and hot-reload inference model."""
    import os

    from app.config import SWIN_UAD_WEIGHTS, WEIGHTS_DIR
    from app.models.swin_backbone import SwinLivenessModel, reload_swin_model

    tensors: list[torch.Tensor] = []
    labels: list[float] = []
    n_real = n_spoof = 0
    for s in samples:
        bgr = _decode_bgr(s.image_base64)
        if bgr is None:
            continue
        tensor, _ = prepare_swin_input(bgr)
        tensors.append(tensor)
        y = 1.0 if int(s.label) == 1 else 0.0
        labels.append(y)
        if y >= 0.5:
            n_spoof += 1
        else:
            n_real += 1

    if n_real < 2 or n_spoof < 2:
        return FinetuneResponse(
            status="skipped_unbalanced",
            n_real=n_real,
            n_spoof=n_spoof,
            error="need at least 2 real and 2 spoof images",
        )
    if len(tensors) < 4:
        return FinetuneResponse(status="skipped_too_small", n_real=n_real, n_spoof=n_spoof)

    x = torch.stack(tensors).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.float32, device=DEVICE)

    model = SwinLivenessModel(pretrained=True).to(DEVICE)
    if SWIN_UAD_WEIGHTS and os.path.isfile(SWIN_UAD_WEIGHTS):
        state = torch.load(SWIN_UAD_WEIGHTS, map_location=DEVICE, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        from app.models.swin_backbone import _load_compatible_state

        _load_compatible_state(model, state, label=SWIN_UAD_WEIGHTS)

    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.uad.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.uad.parameters(), lr=lr)
    crit = torch.nn.BCELoss()
    model.train()
    losses: list[float] = []
    bs = min(8, len(tensors))
    epochs = max(1, min(int(epochs), 10))

    for _ in range(epochs):
        perm = torch.randperm(len(tensors), device=DEVICE)
        for i in range(0, len(tensors), bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            out = model(x[idx])["spoof_score"]
            loss = crit(out, y[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    # Never overwrite calibrated production checkpoints from tiny phone batches.
    out_path = os.path.join(WEIGHTS_DIR, "swin_uad_incremental.pth")
    torch.save(model.state_dict(), out_path)

    # Hot-reload when request asks or env allows (env alone for background trainer).
    env_promote = os.getenv("PROMOTE_INCREMENTAL_WEIGHTS", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )
    do_promote = bool(promote) or env_promote
    if do_promote:
        prod = os.path.join(WEIGHTS_DIR, "swin_uad_finetuned.pth")
        torch.save(model.state_dict(), prod)
        reload_swin_model()
        logger.info("Promoted incremental weights to %s and reloaded", prod)
    else:
        logger.info(
            "UAD incremental finetune saved to %s (not promoted; pass promote=true)",
            out_path,
        )

    mean_loss = float(sum(losses) / len(losses)) if losses else None
    logger.info(
        "UAD finetune done n_real=%s n_spoof=%s epochs=%s mean_loss=%.4f -> %s promote=%s",
        n_real,
        n_spoof,
        epochs,
        mean_loss or -1.0,
        out_path,
        do_promote,
    )
    return FinetuneResponse(
        status="trained",
        n_real=n_real,
        n_spoof=n_spoof,
        epochs=epochs,
        mean_loss=mean_loss,
        weights=out_path if not do_promote else os.path.join(WEIGHTS_DIR, "swin_uad_finetuned.pth"),
        device=str(DEVICE),
        promoted=do_promote,
    )


@router.post("/finetune", response_model=FinetuneResponse)
async def spoof_finetune(req: FinetuneRequest):
    """Incremental UAD fine-tune from labeled phone captures (called by local-api)."""
    import asyncio

    if not req.samples:
        return FinetuneResponse(status="skipped_empty", error="no samples")
    # Cap payload size for safety
    samples = req.samples[:256]
    try:
        return await asyncio.to_thread(
            _run_uad_finetune, samples, req.epochs, req.lr, req.promote
        )
    except Exception as e:
        logger.exception("UAD finetune failed")
        return FinetuneResponse(status="error", error=str(e))
