"""Unified Attack Detection (UAD) module with HiLo self-attention.

HiLo Attention splits heads into:
- Hi (local): window-based attention for high-frequency texture (moire, edges)
- Lo (global): average-pooled attention for low-frequency structure (depth, shape)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoAttention(nn.Module):
    """Low-frequency global attention via average pooling."""

    def __init__(self, dim: int, num_heads: int, pool_size: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pool_size = pool_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        x_pooled = self.pool(x_2d)
        pH, pW = x_pooled.shape[2], x_pooled.shape[3]
        x_flat = x_pooled.flatten(2).transpose(1, 2)

        qkv = self.qkv(x_flat).reshape(B, pH * pW, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, pH * pW, C)
        out = self.proj(out)

        out_2d = out.transpose(1, 2).reshape(B, C, pH, pW)
        out_2d = F.interpolate(out_2d, size=(H, W), mode="bilinear", align_corners=False)
        return out_2d.flatten(2).transpose(1, 2)


class HiAttention(nn.Module):
    """High-frequency local window attention."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, pad_w, 0, pad_h))

        Hp, Wp = x_2d.shape[2], x_2d.shape[3]
        nH, nW = Hp // ws, Wp // ws

        x_win = x_2d.reshape(B, C, nH, ws, nW, ws)
        x_win = x_win.permute(0, 2, 4, 3, 5, 1).reshape(B * nH * nW, ws * ws, C)

        qkv = self.qkv(x_win).reshape(B * nH * nW, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * nH * nW, ws * ws, C)

        out = out.reshape(B, nH, nW, ws, ws, C)
        out = out.permute(0, 5, 1, 3, 2, 4).reshape(B, C, Hp, Wp)
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out.flatten(2).transpose(1, 2)


class HiLoAttentionBlock(nn.Module):
    """Combined HiLo attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hi_ratio: float = 0.625,
        window_size: int = 7,
        pool_size: int = 4,
    ):
        super().__init__()
        hi_heads = int(num_heads * hi_ratio)
        lo_heads = num_heads - hi_heads
        # Ensure at least one head on each path when possible
        if hi_heads == 0 and num_heads > 1:
            hi_heads, lo_heads = 1, num_heads - 1
        if lo_heads == 0 and num_heads > 1:
            lo_heads, hi_heads = 1, num_heads - 1
        hi_dim = hi_heads * (dim // num_heads)
        lo_dim = dim - hi_dim

        self.hi_dim = hi_dim
        self.lo_dim = lo_dim

        self.norm = nn.LayerNorm(dim)
        self.hi_attn = HiAttention(hi_dim, hi_heads, window_size) if hi_heads > 0 else None
        self.lo_attn = LoAttention(lo_dim, lo_heads, pool_size) if lo_heads > 0 else None
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)

        parts = []
        if self.hi_attn is not None:
            parts.append(self.hi_attn(x[:, :, :self.hi_dim], H, W))
        if self.lo_attn is not None:
            parts.append(self.lo_attn(x[:, :, self.hi_dim:], H, W))
        x = torch.cat(parts, dim=-1)

        x = x + shortcut
        x = x + self.ffn(x)
        return x


class UADModule(nn.Module):
    """Unified Attack Detection module.

    Attaches to intermediate features from Swin Stage 3.
    Outputs a spoof probability [0=real, 1=spoof].
    """

    def __init__(self, in_dim: int = 512, num_blocks: int = 2, num_heads: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([
            HiLoAttentionBlock(in_dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, C) feature tokens from Swin Stage 3
            H, W: spatial dimensions

        Returns:
            (B,) spoof probabilities
        """
        for block in self.blocks:
            x = block(x, H, W)

        x = x.mean(dim=1)  # global average pool over tokens
        return self.classifier(x).squeeze(-1)
