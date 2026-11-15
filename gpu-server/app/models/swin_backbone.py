"""Swin Transformer backbone with UAD spoof head and ArcFace FRM head.

Architecture:
- Swin-B pretrained (timm) as shared backbone
- UAD (HiLo) branches from Stage 3 **Block 5** (1-based; index 4) —
  balances local texture with mid-level structure for PAD
- FRM head on final Stage 4 features for 512-D face embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from app.config import DEVICE, SWIN_INPUT_SIZE
from app.models.uad_module import UADModule

# 1-based Block 5 within Swin-B Stage 3 (depth 18)
UAD_STAGE3_BLOCK_1BASED = 5


class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss for discriminative face embeddings."""

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 10000,
        s: float = 64.0,
        m: float = 0.5,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos_theta = nn.functional.linear(
            nn.functional.normalize(embeddings),
            nn.functional.normalize(self.weight),
        )
        theta = torch.acos(cos_theta.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = one_hot * target_logits + (1 - one_hot) * cos_theta
        return output * self.s


class SwinLivenessModel(nn.Module):
    """Unified Swin model with dual heads: UAD (spoof) + FRM (identity)."""

    def __init__(self, pretrained: bool = True, uad_block_1based: int = UAD_STAGE3_BLOCK_1BASED):
        super().__init__()
        self.uad_block_1based = int(uad_block_1based)
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3),  # Stage 3 (full) + Stage 4 for FRM
        )
        stage3_dim = self.backbone.feature_info.channels()[0]
        stage4_dim = self.backbone.feature_info.channels()[1]
        # Slightly more HiLo capacity for phone texture vs spoof moiré
        self.uad = UADModule(in_dim=stage3_dim, num_blocks=3, num_heads=8)

        self.frm_pool = nn.AdaptiveAvgPool2d(1)
        self.frm_proj = nn.Sequential(
            nn.Linear(stage4_dim, 512),
            nn.BatchNorm1d(512),
        )

        self._stage3_blocks = self._resolve_stage3_blocks()
        n_blocks = len(self._stage3_blocks) if self._stage3_blocks is not None else 0
        self._block_index = max(0, min(self.uad_block_1based - 1, max(0, n_blocks - 1)))
        print(
            f"[swin] UAD attached to Stage3 Block {self._block_index + 1}/"
            f"{n_blocks or '?'} (1-based target={self.uad_block_1based})",
            flush=True,
        )

    def _resolve_stage3_blocks(self) -> nn.ModuleList | None:
        """Locate Stage-3 transformer blocks across timm FeatureListNet variants."""
        candidates = []
        root = self.backbone
        # FeatureListNet wraps the original as .model in many timm versions
        for base in (root, getattr(root, "model", None), getattr(root, "pretrained_cfg", None)):
            if base is None or not isinstance(base, nn.Module):
                continue
            candidates.append(base)

        for base in candidates:
            layers = getattr(base, "layers", None)
            if layers is not None and len(layers) >= 3:
                blocks = getattr(layers[2], "blocks", None)
                if blocks is not None and len(blocks) > 0:
                    return blocks
            stages = getattr(base, "stages", None)
            if stages is not None and len(stages) >= 3:
                blocks = getattr(stages[2], "blocks", None)
                if blocks is not None and len(blocks) > 0:
                    return blocks
        return None

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        """Normalize timm Swin feature maps to NCHW."""
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got {tuple(feat.shape)}")
        if feat.shape[1] < feat.shape[-1] and feat.shape[-1] in (128, 256, 512, 1024):
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _tokens_from_block_out(self, out: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """Convert a Swin block output to (B, H*W, C) tokens + spatial size."""
        if out.dim() == 3:
            # (B, N, C) — infer H,W as square when possible
            b, n, c = out.shape
            side = int(n**0.5)
            if side * side == n:
                return out, side, side
            # Fallback reshape guess used by some Swin variants
            return out, side, max(1, n // max(side, 1))
        feat = self._to_nchw(out)
        _, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, h, w

    def _stage_tokens(self, feat: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        feat = self._to_nchw(feat)
        _, _, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        return tokens, h, w

    def _forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, torch.Tensor]:
        """Return (uad_tokens, H, W, stage4_nchw). Prefers Stage3 Block 5 via hook."""
        block5_out: dict[str, torch.Tensor] = {}
        handle = None
        if self._stage3_blocks is not None and len(self._stage3_blocks) > 0:
            idx = self._block_index

            def _hook(_module, _inp, out):
                block5_out["feat"] = out

            handle = self._stage3_blocks[idx].register_forward_hook(_hook)

        try:
            features = self.backbone(x)
        finally:
            if handle is not None:
                handle.remove()

        if "feat" in block5_out:
            tokens, h, w = self._tokens_from_block_out(block5_out["feat"])
        else:
            # Fallback: end of Stage 3 (previous behaviour)
            tokens, h, w = self._stage_tokens(features[0])

        stage4 = self._to_nchw(features[1])
        return tokens, h, w, stage4

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) input images

        Returns:
            dict with spoof_score (B,) and embedding (B, 512)
        """
        tokens, h, w, stage4 = self._forward_features(x)
        spoof_score = self.uad(tokens, h, w)

        pooled = self.frm_pool(stage4).flatten(1)
        embedding = self.frm_proj(pooled)
        embedding = nn.functional.normalize(embedding, dim=1)

        return {"spoof_score": spoof_score, "embedding": embedding}

    def get_spoof_score(self, x: torch.Tensor) -> torch.Tensor:
        tokens, h, w, _ = self._forward_features(x)
        return self.uad(tokens, h, w)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, stage4 = self._forward_features(x)
        pooled = self.frm_pool(stage4).flatten(1)
        embedding = self.frm_proj(pooled)
        return nn.functional.normalize(embedding, dim=1)


_model: SwinLivenessModel | None = None


def _load_compatible_state(model: nn.Module, state: dict, label: str = "checkpoint") -> tuple[int, int]:
    """Load matching tensors only; skip size-mismatched keys (architecture upgrades)."""
    current = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state.items():
        if k not in current:
            skipped += 1
            continue
        if tuple(current[k].shape) != tuple(v.shape):
            skipped += 1
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"[swin] {label}: loaded={len(filtered)} skipped={skipped} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return len(filtered), skipped


def _load_weights(model: SwinLivenessModel) -> None:
    from app.config import SWIN_UAD_WEIGHTS
    import os

    path = SWIN_UAD_WEIGHTS
    if not path or not os.path.isfile(path):
        print(f"[swin] No checkpoint at {path!r} — using ImageNet backbone + random UAD head")
        return
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    _load_compatible_state(model, state, label=path)


def get_swin_model() -> SwinLivenessModel:
    global _model
    if _model is None:
        _model = SwinLivenessModel(pretrained=True).to(DEVICE)
        _load_weights(_model)
        _model.eval()
    return _model


def reload_swin_model() -> SwinLivenessModel:
    """Drop cached weights and reload from disk (after finetune)."""
    global _model
    _model = None
    return get_swin_model()
