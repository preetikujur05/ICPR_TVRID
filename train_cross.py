# train_cross.py
# ─────────────────────────────────────────────────────────────────────────────
# Track 3 – Cross-modal RGB-D Fusion Re-ID training
#
#   python train_cross.py
#   python train_cross.py --epochs 25 --batch_size 6 --lr 3e-4
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L

from models.cross_model import FusionReID
from utils.data import DataConfig, UnifiedReIDDataModule


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (mirror config/train.yaml  →  cross section)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data_root="data",
    train_csv="data/train_split.csv",
    eval_csv="data/valid_split.csv",
    modality="rgbd",
    epochs=20,
    batch_size=6,
    embedding_size=256,
    lr=3e-4,
    margin=0.3,
    checkpoint="checkpoints/fusion_cross.ckpt",
    num_workers=0,
    accelerator="gpu",
    devices=1,
    precision=16,
)


# ─────────────────────────────────────────────────────────────────────────────
# Training logic
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    Path("checkpoints").mkdir(exist_ok=True)

    # ── DataModule ────────────────────────────────────────────────────────────
    data_cfg = DataConfig(
        root=args.data_root,
        train_csv=args.train_csv,
        eval_csv=args.eval_csv,
        modality=args.modality,         # "rgbd"
        val_mode="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=False,
    )
    data_cfg.sequence.length = 1        # single-frame training

    dm = UnifiedReIDDataModule(data_cfg)
    dm.setup("fit")

    # ── Model ─────────────────────────────────────────────────────────────────
    num_classes = dm.train_set.df["person_id"].nunique()
    print(f"[Cross] num_classes = {num_classes}")

    model = FusionReID(
        num_classes=num_classes,
        embedding_size=args.embedding_size,
        lr=args.lr,
        margin=args.margin,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        precision=args.precision,
        log_every_n_steps=10,
        enable_checkpointing=True,
        default_root_dir="lightning_logs",
    )

    trainer.fit(model, datamodule=dm)
    trainer.save_checkpoint(args.checkpoint)
    print(f"[Cross] Checkpoint saved → {args.checkpoint}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Cross-modal Fusion Re-ID model (Track 3)"
    )
    p.add_argument("--data_root",      default=DEFAULTS["data_root"])
    p.add_argument("--train_csv",      default=DEFAULTS["train_csv"])
    p.add_argument("--eval_csv",       default=DEFAULTS["eval_csv"])
    p.add_argument("--modality",       default=DEFAULTS["modality"])
    p.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--embedding_size", type=int,   default=DEFAULTS["embedding_size"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--margin",         type=float, default=DEFAULTS["margin"])
    p.add_argument("--checkpoint",     default=DEFAULTS["checkpoint"])
    p.add_argument("--num_workers",    type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--accelerator",    default=DEFAULTS["accelerator"])
    p.add_argument("--devices",        type=int,   default=DEFAULTS["devices"])
    p.add_argument("--precision",      type=int,   default=DEFAULTS["precision"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
