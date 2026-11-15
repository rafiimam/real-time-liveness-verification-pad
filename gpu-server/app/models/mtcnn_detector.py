"""MTCNN face detection and alignment using facenet-pytorch."""

import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from app.config import DEVICE, FACE_SIZE


class FaceDetector:
    def __init__(self, min_face_size: int = 40, thresholds: list | None = None):
        self.min_face_size = min_face_size
        self.mtcnn = MTCNN(
            image_size=FACE_SIZE,
            margin=20,
            min_face_size=min_face_size,
            thresholds=thresholds or [0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=DEVICE,
            keep_all=True,
        )

    def detect_and_align(self, image: np.ndarray, min_prob: float = 0.9) -> list[dict]:
        """Detect faces and return aligned crops with bounding boxes.

        Args:
            image: BGR numpy array (H, W, 3)

        Returns:
            List of dicts with keys: crop (112x112 tensor), box, confidence, landmarks
        """
        rgb = image[:, :, ::-1]
        pil_img = Image.fromarray(rgb)

        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        if boxes is None:
            return []

        faces = self.mtcnn.extract(pil_img, boxes, save_path=None)
        if faces is None:
            return []

        results = []
        for i, (box, prob, lm, face) in enumerate(zip(boxes, probs, landmarks, faces)):
            if prob is None or float(prob) < min_prob:
                continue
            results.append({
                "crop": face,
                "box": box.tolist(),
                "confidence": float(prob),
                "landmarks": lm.tolist() if lm is not None else None,
            })
        return results

    def get_best_face(self, image: np.ndarray) -> torch.Tensor | None:
        """Return the highest-confidence aligned face crop as a tensor."""
        faces = self.detect_and_align(image)
        if not faces:
            return None
        best = max(faces, key=lambda f: f["confidence"])
        return best["crop"]


_detector: FaceDetector | None = None
_doc_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector(min_face_size=40)
    return _detector


def get_document_detector() -> FaceDetector:
    """Lower min_face_size for scanned signature cards / ID portraits."""
    global _doc_detector
    if _doc_detector is None:
        _doc_detector = FaceDetector(
            min_face_size=20,
            thresholds=[0.5, 0.6, 0.6],
        )
    return _doc_detector
