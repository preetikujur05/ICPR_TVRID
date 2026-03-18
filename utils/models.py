# utils/models.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared model building-blocks for all three ICPR-2026-TVRID tracks.
#
#   • _ensure_sequence()        – shape normaliser (B,C,H,W → B,1,C,H,W)
#   • TripletLoss               – margin-based triplet loss
#   • ConvNeXtRGBEncoder        – ConvNeXt-Tiny backbone for RGB (3-ch)
#   • ConvNeXtDepthEncoder      – ConvNeXt-Tiny backbone for Depth (1-ch)
#
# Track-specific Lightning modules live in their own files:
#   models/rgb_model.py   → RGBReIDLightning
#   models/depth_model.py → DepthOnlyTrainer
#   models/cross_model.py → FusionReID
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    """
    Normalise tensor to shape (B, S, C, H, W).
    Single-frame tensors (B, C, H, W) are promoted to S=1.
    """
    if x.ndim == 4:
        return x.unsqueeze(1)   # B,1,C,H,W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Squared-Euclidean triplet loss with soft margin.

    Args:
        margin: Minimum required gap between d(a,p) and d(a,n).
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        d_ap = (anchor - positive).pow(2).sum(dim=1)
        d_an = (anchor - negative).pow(2).sum(dim=1)
        return torch.relu(d_ap - d_an + self.margin).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

class ConvNeXtRGBEncoder(nn.Module):
    """
    ConvNeXt-Tiny encoder for **3-channel RGB** images.

    Input  : (B, [S,] 3, H, W)
    Output : (B, embedding_size)  – L2-normalised
    """

    def __init__(self, embedding_size: int = 256, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.pool(self.backbone(x))
        emb = self.head(feats)
        emb = emb.view(B, S, -1).mean(dim=1)
        return F.normalize(emb, dim=1)


class ConvNeXtDepthEncoder(nn.Module):
    """
    ConvNeXt-Tiny encoder for **1-channel Depth** images.

    The first convolutional layer is replaced to accept single-channel input
    while retaining all other pretrained weights.

    Input  : (B, [S,] 1, H, W)
    Output : (B, embedding_size)  – L2-normalised
    """

    def __init__(self, embedding_size: int = 256) -> None:
        super().__init__()
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Replace first Conv2d (3-ch → 1-ch) in the stem block
        stem_block = convnext.features[0]   # nn.Sequential
        first = stem_block[0]               # nn.Conv2d
        stem_block[0] = nn.Conv2d(
            1,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        self.backbone = convnext.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.pool(self.backbone(x))
        emb = self.head(feats)
        emb = emb.view(B, S, -1).mean(dim=1)
        return F.normalize(emb, dim=1)
