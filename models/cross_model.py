# models/cross_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Track 3 – Cross-modal RGB-D Fusion Re-ID
#
#   FusionEncoder   – concatenates RGB and Depth embeddings, projects to emb.
#   FusionReID      – LightningModule  (triplet + CE loss, AdamW)
#
# Query  : RGB image
# Gallery: Depth image
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from utils.models import ConvNeXtDepthEncoder, ConvNeXtRGBEncoder, TripletLoss


# ─────────────────────────────────────────────────────────────────────────────
# Fusion encoder
# ─────────────────────────────────────────────────────────────────────────────

class FusionEncoder(nn.Module):
    """
    Fuses RGB and Depth streams into a single embedding.

    Architecture:
        rgb_emb   = ConvNeXtRGBEncoder(rgb)       # (B, E)
        depth_emb = ConvNeXtDepthEncoder(depth)    # (B, E)
        fused     = MLP(concat([rgb_emb, depth_emb]))  # (B, E)

    Args:
        embedding_size: Dimensionality E of each stream and the fused output.
    """

    def __init__(self, embedding_size: int = 256) -> None:
        super().__init__()
        self.rgb_enc = ConvNeXtRGBEncoder(embedding_size)
        self.depth_enc = ConvNeXtDepthEncoder(embedding_size)

        self.fusion = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        fr = self.rgb_enc(rgb)
        fd = self.depth_enc(depth)
        return self.fusion(torch.cat([fr, fd], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class FusionReID(L.LightningModule):
    """
    Lightning module for the **cross-modal RGB-D Fusion** Re-ID track.

    Training triplet setup:
        anchor   = RGB  image
        positive = Depth image (same identity)
        negative = Depth image (different identity)

    Loss:
        total = ce_loss(anchor_logits, labels) + triplet_loss(ea, ep, en)

    Args:
        num_classes    : Number of training identities.
        embedding_size : Embedding dimensionality (default 256).
        lr             : AdamW learning rate (default 3e-4).
        margin         : Triplet-loss margin (default 0.3).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 256,
        lr: float = 3e-4,
        margin: float = 0.3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = FusionEncoder(embedding_size)

        # BNNeck + classifier
        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

        self.triplet = TripletLoss(margin)
        self.ce = nn.CrossEntropyLoss()

    # ── Forward / encode ──────────────────────────────────────────────────────

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.encoder(rgb, depth)

    def encode(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Public alias used during evaluation embedding loops."""
        return self.encoder(rgb, depth)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        labels = batch["person_id"].long()

        ea = self(batch["anchor"]["rgb"],   batch["anchor"]["depth"])
        ep = self(batch["positive"]["rgb"], batch["positive"]["depth"])
        en = self(batch["negative"]["rgb"], batch["negative"]["depth"])

        logits = self.classifier(self.bnneck(ea))
        loss = self.ce(logits, labels) + self.triplet(ea, ep, en)

        self.log("train/loss", loss, prog_bar=True)
        return loss

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
