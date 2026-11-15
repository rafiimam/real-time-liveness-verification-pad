"""Process full 10s video stream endpoint."""

import base64
import io
import tempfile

import cv2
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel

from app.config import DEVICE, SWIN_INPUT_SIZE
from app.models.mtcnn_detector import get_detector
from app.models.swin_backbone import get_swin_model
from app.models.arcface_head import get_embedder
from app.models.temporal_3dcnn import get_temporal_model, CHUNK_FRAMES

router = APIRouter()


class StreamResponse(BaseModel):
    total_frames: int
    faces_detected: int
    spatial_spoof_scores: list[float]
    temporal_chunk_scores: list[float]
    mean_spatial_spoof: float
    mean_temporal_liveness: float
    best_frame_embedding: list[float]
    is_live: bool


@router.post("", response_model=StreamResponse)
async def process_stream(video: UploadFile = File(...)):
    """Process a full 10s video and return aggregated liveness + embedding."""
    content = await video.read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return StreamResponse(
            total_frames=0, faces_detected=0,
            spatial_spoof_scores=[], temporal_chunk_scores=[],
            mean_spatial_spoof=1.0, mean_temporal_liveness=0.0,
            best_frame_embedding=[], is_live=False,
        )

    detector = get_detector()
    swin_model = get_swin_model()
    embedder = get_embedder()
    temporal_model = get_temporal_model()

    # Detect faces and collect aligned crops
    crops_for_spatial = []
    crops_for_temporal = []
    best_quality = -1.0
    best_frame_idx = 0
    faces_detected = 0

    for i, frame in enumerate(frames):
        faces = detector.detect_and_align(frame)
        if not faces:
            continue
        faces_detected += 1
        best_face = max(faces, key=lambda f: f["confidence"])
        crop = best_face["crop"]  # (3, 112, 112) tensor

        # Quality: use confidence as proxy
        if best_face["confidence"] > best_quality:
            best_quality = best_face["confidence"]
            best_frame_idx = i

        # Resize for Swin (224x224)
        crop_np = ((crop.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        crop_resized = cv2.resize(crop_np, (SWIN_INPUT_SIZE, SWIN_INPUT_SIZE))
        tensor_224 = torch.from_numpy(crop_resized).float().permute(2, 0, 1) / 255.0
        tensor_224 = (tensor_224 - 0.5) / 0.5
        crops_for_spatial.append(tensor_224)

        # For temporal: 112x112
        crops_for_temporal.append(crop_np.astype(np.float32) / 255.0)

    # Spatial spoof scoring
    spatial_scores = []
    if crops_for_spatial:
        batch = torch.stack(crops_for_spatial).to(DEVICE)
        with torch.no_grad():
            for i in range(0, len(batch), 16):
                chunk = batch[i:i+16]
                scores = swin_model.get_spoof_score(chunk)
                spatial_scores.extend(scores.cpu().tolist())

    # Temporal scoring
    temporal_scores = []
    if crops_for_temporal:
        arr = np.array(crops_for_temporal)
        for i in range(0, len(arr), CHUNK_FRAMES):
            chunk = arr[i:i + CHUNK_FRAMES]
            if len(chunk) < CHUNK_FRAMES:
                pad = np.zeros((CHUNK_FRAMES - len(chunk), *chunk.shape[1:]), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])
            tensor = torch.from_numpy(chunk).permute(3, 0, 1, 2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                score = temporal_model(tensor)
            temporal_scores.append(float(score[0]))

    # Best frame embedding
    best_emb = embedder.get_embedding_from_frame(frames[best_frame_idx])
    emb_list = best_emb.tolist() if best_emb is not None else []

    mean_spatial = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 1.0
    mean_temporal_fake = sum(temporal_scores) / len(temporal_scores) if temporal_scores else 1.0
    mean_temporal_live = 1.0 - mean_temporal_fake

    return StreamResponse(
        total_frames=len(frames),
        faces_detected=faces_detected,
        spatial_spoof_scores=spatial_scores,
        temporal_chunk_scores=temporal_scores,
        mean_spatial_spoof=mean_spatial,
        mean_temporal_liveness=mean_temporal_live,
        best_frame_embedding=emb_list,
        is_live=(mean_spatial < 0.5 and mean_temporal_live > 0.9),
    )
