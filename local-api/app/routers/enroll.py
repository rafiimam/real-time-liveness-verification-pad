"""Faceprint enrollment and account management."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.db.pgvector import store_faceprint, get_reference_image_b64
from app.security.auth import require_admin, require_api_auth
from app.services import gpu_client

router = APIRouter()


class EnrollRequest(BaseModel):
    account_id: str
    image_base64: str
    embedding: Optional[list[float]] = None


class EnrollResponse(BaseModel):
    account_id: str
    enrolled: bool
    embedding_dim: int
    message: str


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_faceprint(
    req: EnrollRequest,
    _auth: dict = Depends(require_admin),
):
    """Register an account holder's reference face image (admin only)."""
    embedding = req.embedding
    if embedding is None:
        import base64
        import cv2
        import numpy as np

        img_bytes = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        embedding = await gpu_client.get_embedding(image)
        if embedding is None:
            raise HTTPException(status_code=400, detail="No face detected in reference image")

    await store_faceprint(
        account_id=req.account_id,
        embedding=embedding,
        reference_image_b64=req.image_base64,
    )

    return EnrollResponse(
        account_id=req.account_id,
        enrolled=True,
        embedding_dim=len(embedding),
        message="Faceprint enrolled successfully",
    )


@router.get("/{account_id}/reference")
async def check_reference(
    account_id: str,
    _auth: dict = Depends(require_api_auth),
):
    """Check whether an account has a registered reference face."""
    ref = await get_reference_image_b64(account_id)
    return {
        "account_id": account_id,
        "has_reference": ref is not None,
    }
