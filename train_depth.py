# train_depth.py
# ─────────────────────────────────────────────────────────────────────────────
# Track 2 – Depth-only Re-ID training
#
#   python train_depth.py
#   python train_depth.py --epochs 35 --batch_size 6 --lr 1e-4
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch

from models.depth_model import DepthOnlyTrainer
from utils.data import DataConfig, UnifiedReIDDataModule


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (mirror config/train.yaml  →  depth section)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    data_root="data",
    train_csv="data/train_split.csv",
    eval_csv="data/valid_split.csv",
    modality="depth",
    epochs=35,
    batch_size=6,
    embedding_size=256,
    lr=1e-4,
    margin=0.5,
    checkpoint="checkpoints/best_depth.ckpt",
    weights_pt="checkpoints/best_depth_pure.pt",
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
        modality=args.modality,         # "depth"
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
    print(f"[Depth] num_classes = {num_classes}")

    model = DepthOnlyTrainer(
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

    # Save both Lightning checkpoint and raw state-dict
    trainer.save_checkpoint(args.checkpoint)
    torch.save(model.state_dict(), args.weights_pt)
    print(f"[Depth] Lightning checkpoint → {args.checkpoint}")
    print(f"[Depth] State-dict           → {args.weights_pt}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Depth Re-ID model (Track 2)")
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
    p.add_argument("--weights_pt",     default=DEFAULTS["weights_pt"])
    p.add_argument("--num_workers",    type=int,   default=DEFAULTS["num_workers"])
    p.add_argument("--accelerator",    default=DEFAULTS["accelerator"])
    p.add_argument("--devices",        type=int,   default=DEFAULTS["devices"])
    p.add_argument("--precision",      type=int,   default=DEFAULTS["precision"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
