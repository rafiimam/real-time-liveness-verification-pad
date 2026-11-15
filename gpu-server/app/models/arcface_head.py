"""ArcFace embedding extraction using InsightFace pretrained model."""

import numpy as np
import torch
from insightface.app import FaceAnalysis

from app.config import DEVICE


class ArcFaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))

    def get_embedding_from_frame(self, image: np.ndarray) -> np.ndarray | None:
        """Extract 512-D embedding from a full frame (uses internal detection).

        Args:
            image: BGR numpy array

        Returns:
            512-D normalized embedding or None if no face found
        """
        faces = self.app.get(image)
        if not faces:
            return None
        best = max(faces, key=lambda f: f.det_score)
        return best.normed_embedding

    def get_embedding_from_crop(self, crop_tensor: torch.Tensor) -> np.ndarray:
        """Extract embedding from an already-aligned 112x112 face crop tensor.

        The crop tensor is expected in [-1, 1] range, shape (3, 112, 112).
        Uses the recognition model directly.
        """
        img = ((crop_tensor.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
        img_bgr = img[:, :, ::-1].copy()
        faces = self.app.get(img_bgr)
        if not faces:
            return np.zeros(512, dtype=np.float32)
        return faces[0].normed_embedding

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


_embedder: ArcFaceEmbedder | None = None


def get_embedder() -> ArcFaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ArcFaceEmbedder()
    return _embedder
