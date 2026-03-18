# models/rgb_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Track 1 – RGB-only Re-ID
#
#   RGBReIDLightning
#     • Backbone : ConvNeXtRGBEncoder  (ConvNeXt-Tiny, pretrained)
#     • Loss     : TripletLoss + CrossEntropyLoss (BNNeck)
#     • Optimiser: AdamW + CosineAnnealingLR
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn

from utils.models import ConvNeXtDepthEncoder, ConvNeXtRGBEncoder, TripletLoss


class RGBReIDLightning(L.LightningModule):
    """
    Lightning module for the **RGB-only** Re-ID track.

    Args:
        num_classes     : Number of distinct identities in the training split.
        embedding_size  : Dimensionality of the output embedding (default 256).
        lr              : Initial learning rate (default 3e-4).
        margin          : Triplet-loss margin (default 0.3).
        anchor_modality : Modality key for the anchor sample   (default "rgb").
        positive_modality: Modality key for the positive sample (default "rgb").
        negative_modality: Modality key for the negative sample (default "rgb").
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int = 256,
        lr: float = 3e-4,
        margin: float = 0.3,
        anchor_modality: str = "rgb",
        positive_modality: str = "rgb",
        negative_modality: str = "rgb",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Encoders (both kept so a checkpoint can be used for cross-modal eval)
        self.rgb_encoder = ConvNeXtRGBEncoder(embedding_size)
        self.depth_encoder = ConvNeXtDepthEncoder(embedding_size)

        # Classification head (BNNeck pattern)
        self.bnneck = nn.BatchNorm1d(embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

        # Losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin)

    # ── Encoder dispatch ──────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Return an L2-normalised embedding for the given modality tensor."""
        if modality == "rgb":
            return self.rgb_encoder(x)
        if modality == "depth":
            return self.depth_encoder(x)
        raise ValueError(f"Unknown modality: '{modality}'")

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        labels = batch["person_id"].long()

        a = batch["anchor"][self.hparams.anchor_modality]
        p = batch["positive"][self.hparams.positive_modality]
        n = batch["negative"][self.hparams.negative_modality]

        ea = self.encode(a, self.hparams.anchor_modality)
        ep = self.encode(p, self.hparams.positive_modality)
        en = self.encode(n, self.hparams.negative_modality)

        # CE loss on anchor embedding via BNNeck
        logits = self.classifier(self.bnneck(ea))
        loss_ce = self.ce_loss(logits, labels)
        loss_tri = self.triplet_loss(ea, ep, en)
        loss = loss_ce + loss_tri

        self.log_dict(
            {"train/loss": loss, "train/ce": loss_ce, "train/triplet": loss_tri},
            prog_bar=True,
        )
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx: int) -> None:
        if not {"anchor", "positive", "negative"} <= set(batch.keys()):
            return

        a = batch["anchor"][self.hparams.anchor_modality]
        p = batch["positive"][self.hparams.positive_modality]
        n = batch["negative"][self.hparams.negative_modality]

        ea = self.encode(a, self.hparams.anchor_modality)
        ep = self.encode(p, self.hparams.positive_modality)
        en = self.encode(n, self.hparams.negative_modality)

        loss = self.triplet_loss(ea, ep, en)
        d_ap = (ea - ep).pow(2).sum(1)
        d_an = (ea - en).pow(2).sum(1)
        acc = (d_ap < d_an).float().mean()

        self.log_dict(
            {"val/loss": loss, "val/accuracy": acc},
            prog_bar=True,
            sync_dist=True,
        )

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=5e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
        return {"optimizer": opt, "lr_scheduler": sch}
