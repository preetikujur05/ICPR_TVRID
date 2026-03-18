# prepare_data.py
# ─────────────────────────────────────────────────────────────────────────────
# Splits the raw training CSV into identity-disjoint train and validation sets.
# Run this **once** before training any track.
#
#   python prepare_data.py
#
# Outputs:
#   data/train_split.csv   – training identities
#   data/valid_split.csv   – validation identities
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_splits(
    input_csv: str,
    train_out: str,
    val_out: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """
    Create identity-disjoint train / validation CSV files.

    Args:
        input_csv : Path to the original train_labels.csv.
        train_out : Destination path for the training split.
        val_out   : Destination path for the validation split.
        val_ratio : Fraction of identities assigned to validation.
        seed      : Random seed for reproducibility.
    """
    df = pd.read_csv(input_csv)

    person_ids = df["person_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(person_ids)

    n_val = int(len(person_ids) * val_ratio)
    val_ids = set(person_ids[:n_val])
    train_ids = set(person_ids[n_val:])

    train_df = df[df["person_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_ids)].reset_index(drop=True)

    Path(train_out).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    print(
        f"Train split → {train_out}  "
        f"({train_df['person_id'].nunique()} identities, {len(train_df)} rows)"
    )
    print(
        f"Valid split → {val_out}  "
        f"({val_df['person_id'].nunique()} identities, {len(val_df)} rows)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create train/val CSV splits")
    p.add_argument("--input_csv",  default="data/train_labels.csv")
    p.add_argument("--train_out",  default="data/train_split.csv")
    p.add_argument("--val_out",    default="data/valid_split.csv")
    p.add_argument("--val_ratio",  type=float, default=0.2)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_splits(
        input_csv=args.input_csv,
        train_out=args.train_out,
        val_out=args.val_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
