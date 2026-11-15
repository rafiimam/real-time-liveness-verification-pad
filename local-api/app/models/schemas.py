"""Pydantic schemas for the decision API."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SessionStatus(str, Enum):
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CreateSessionRequest(BaseModel):
    account_id: str
    transaction_id: Optional[str] = None
    transaction_amount: Optional[float] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    nonce: str
    created_at: datetime


class VerificationResult(BaseModel):
    session_id: str
    status: SessionStatus
    face_match: Optional[bool] = None
    face_match_confidence: Optional[float] = None
    nid_photo_match: Optional[bool] = None
    nid_photo_match_confidence: Optional[float] = None
    account_photo_match: Optional[bool] = None
    account_photo_match_confidence: Optional[float] = None
    nid_verified: Optional[bool] = None
    customer_name: Optional[str] = None
    face_match_provider: Optional[str] = None  # polygon | azure | internal
    cbs_faces_detected: Optional[int] = None
    liveness: Optional[bool] = None
    spatial_liveness_score: Optional[float] = None
    temporal_liveness_score: Optional[float] = None
    screen_spoof_score: Optional[float] = None
    composite_risk_score: Optional[float] = None
    is_approved: Optional[bool] = None
    reason_codes: Optional[list[str]] = None
    capture_trust: Optional[str] = None  # live_ws | upload | unknown
    challenge: Optional[dict] = None  # step-up: {code, prompt}
    completed_at: Optional[datetime] = None


class GPUStreamResult(BaseModel):
    total_frames: int
    faces_detected: int
    spatial_spoof_scores: list[float]
    temporal_chunk_scores: list[float]
    mean_spatial_spoof: float
    mean_temporal_liveness: float
    best_frame_embedding: list[float]
    is_live: bool
