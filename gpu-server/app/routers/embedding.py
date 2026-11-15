"""Face embedding endpoint."""

import base64

import cv2
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.arcface_head import get_embedder

router = APIRouter()


class EmbeddingRequest(BaseModel):
    image_base64: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimension: int


@router.post("", response_model=EmbeddingResponse)
async def get_embedding(req: EmbeddingRequest):
    img_bytes = base64.b64decode(req.image_base64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    embedder = get_embedder()
    emb = embedder.get_embedding_from_frame(image)

    if emb is None:
        return EmbeddingResponse(embedding=[], dimension=0)

    return EmbeddingResponse(embedding=emb.tolist(), dimension=len(emb))
