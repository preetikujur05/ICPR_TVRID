# models/depth_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Track 2 – Depth-only Re-ID
#
#   DepthPreprocessor
#     • Converts a raw 1-ch depth map to a richer 3-ch representation:
#         ch0 – normalised depth
#         ch1 – Sobel gradient magnitude
#         ch2 – local average-pooled context
#
#   DepthOnlyTrainer  (LightningModule)
#     • Backbone : ConvNeXtDepthEncoder  (1-ch ConvNeXt-Tiny, pretrained)
#     • Loss     : TripletLoss (margin=0.5) + 0.5 × CrossEntropyLoss
#     • Optimiser: AdamW (lr=1e-4)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import cv2
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models import ConvNeXtDepthEncoder, TripletLoss


# ─────────────────────────────────────────────────────────────────────────────
# Depth pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

class DepthPreprocessor:
    """
    Converts a raw depth image path into a 3-channel tensor suitable for
    ConvNeXtDepthEncoder.

    Usage::

        preprocessor = DepthPreprocessor(size=224)
        tensor = preprocessor(depth_path)   # shape (3, 224, 224)
    """

    def __init__(self, size: int = 224) -> None:
        self.size = size

    def load(self, path: str) -> np.ndarray:
        """Read a depth image via OpenCV and resize to (size, size)."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (self.size, self.size))
        return img.astype(np.float32)

    def to_3ch(self, depth_img: np.ndarray) -> torch.Tensor:
        """
        Encode a (H, W) float32 depth array as a 3-channel tensor:
          ch0 – normalised raw depth
          ch1 – Sobel gradient magnitude (edge map)
          ch2 – local neighbourhood average (context)
        """
        depth = torch.tensor(depth_img).float().unsqueeze(0)    # (1,H,W)
        depth = depth / (depth.max() + 1e-6)

        sobel_x = torch.abs(torch.diff(depth, dim=1, prepend=depth[:, :1, :]))
        sobel_y = torch.abs(torch.diff(depth, dim=2, prepend=depth[:, :, :1]))
        ch1 = (sobel_x + sobel_y) / (sobel_x.max() + 1e-6)

        ch2 = F.avg_pool2d(
            depth.unsqueeze(0), kernel_size=5, stride=1, padding=2
        ).squeeze(0)

        return torch.cat([depth, ch1, ch2], dim=0)              # (3,H,W)

    def __call__(self, path: str) -> torch.Tensor:
        return self.to_3ch(self.load(path))


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class DepthOnlyTrainer(L.LightningModule):
    """
    Lightning module for the **Depth-only** Re-ID track.

    Architecture:
        • ConvNeXtDepthEncoder (1-channel input, 256-dim output)
        • Linear classifier head for auxiliary CE supervision

    Loss:
        total = triplet_loss + 0.5 × ce_loss

    Args:
        num_classes    : Number of training identities.
        embedding_size : Embedding dimensionality (default 256).
        lr             : AdamW learning rate (default 1e-4).
        margin         : Triplet-loss margin (default 0.5).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 256,
        lr: float = 1e-4,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ConvNeXtDepthEncoder(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

        self.triplet = TripletLoss(margin)
        self.ce_loss = nn.CrossEntropyLoss()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised depth embedding."""
        return self.encoder(x)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        a = batch["anchor"]["depth"]
        p = batch["positive"]["depth"]
        n = batch["negative"]["depth"]
        labels = batch["person_id"].long()

        ea, ep, en = self(a), self(p), self(n)

        loss_tri = self.triplet(ea, ep, en)
        loss_ce = self.ce_loss(self.classifier(ea), labels)
        loss = loss_tri + 0.5 * loss_ce

        self.log_dict(
            {
                "depth/loss": loss,
                "depth/triplet": loss_tri,
                "depth/ce": loss_ce,
            },
            prog_bar=True,
        )
        return loss

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )
