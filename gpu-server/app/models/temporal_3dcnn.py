"""3D ResNet for temporal liveness detection (FakeBuster-style).

Processes 30-frame video chunks and predicts a fakeness score.
Uses torchvision's r3d_18 as backbone, fine-tuned for binary spoof detection.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from app.config import DEVICE

CHUNK_FRAMES = 30


class TemporalLivenessModel(nn.Module):
    """3D ResNet-18 adapted for binary liveness classification.

    Input: (B, 3, T=30, H=112, W=112) video tensor
    Output: (B,) fakeness probability [0=real, 1=fake]
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        base = r3d_18(weights=weights)

        self.stem = base.stem
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 30, 112, 112)

        Returns:
            (B,) fakeness scores
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(-1)


_model: TemporalLivenessModel | None = None


def _load_weights(model: TemporalLivenessModel) -> None:
    from app.config import TEMPORAL_WEIGHTS
    import os

    path = TEMPORAL_WEIGHTS
    if not path or not os.path.isfile(path):
        print(f"[temporal] No checkpoint at {path!r} — using Kinetics backbone + random head")
        return
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[temporal] Loaded weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})")


def get_temporal_model() -> TemporalLivenessModel:
    global _model
    if _model is None:
        _model = TemporalLivenessModel(pretrained=True).to(DEVICE)
        _load_weights(_model)
        _model.eval()
    return _model
