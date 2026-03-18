# eval_generate.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared evaluation script – generates ranked gallery CSV for all tracks.
#
# Usage:
#   python eval_generate.py --track rgb
#   python eval_generate.py --track depth
#   python eval_generate.py --track cross
#   python eval_generate.py --track all      # run all three sequentially
#
# Output:
#   outputs/rankings_rgb.csv
#   outputs/rankings_depth.csv
#   outputs/rankings_cross.csv
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from models.cross_model import FusionReID
from models.depth_model import DepthOnlyTrainer
from models.rgb_model import RGBReIDLightning
from utils.data import DataConfig, UnifiedReIDDataset, build_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Track registry
# ─────────────────────────────────────────────────────────────────────────────

TRACK_CONFIGS: Dict[str, dict] = {
    "rgb": {
        "data_modality": "rgb",
        "query_mod": "rgb",
        "gallery_mod": "rgb",
        "checkpoint": "checkpoints/best_rgb.ckpt",
        "output": "outputs/rankings_rgb.csv",
        "model_cls": RGBReIDLightning,
        "load_fn": "lightning",         # use load_from_checkpoint
    },
    "depth": {
        "data_modality": "depth",
        "query_mod": "depth",
        "gallery_mod": "depth",
        "checkpoint": "checkpoints/best_depth_pure.pt",
        "output": "outputs/rankings_depth.csv",
        "model_cls": DepthOnlyTrainer,
        "load_fn": "state_dict",        # use load_state_dict from .pt
    },
    "cross": {
        "data_modality": "rgbd",
        "query_mod": "rgb",
        "gallery_mod": "depth",
        "checkpoint": "checkpoints/fusion_cross.ckpt",
        "output": "outputs/rankings_cross.csv",
        "model_cls": FusionReID,
        "load_fn": "lightning",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(cfg: dict, train_csv: str, device: str):
    """Load the correct model class from its checkpoint."""
    ckpt = cfg["checkpoint"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    model_cls = cfg["model_cls"]

    if cfg["load_fn"] == "lightning":
        model = model_cls.load_from_checkpoint(ckpt)

    else:   # state_dict (.pt) – DepthOnlyTrainer needs num_classes
        import pandas as pd
        num_classes = pd.read_csv(train_csv)["person_id"].nunique()
        model = model_cls(lr=1e-4, margin=0.5, num_classes=num_classes)
        model.load_state_dict(torch.load(ckpt, map_location=device))

    return model.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_embeddings(
    loader,
    model,
    track_cfg: dict,
    device: str,
    track_name: str,
):
    """
    Run inference over *loader* and return (ids, paths, query_embs, gallery_embs).
    For the cross track query=RGB, gallery=Depth.
    For rgb/depth both query and gallery are the same modality.
    """
    ids: List[str] = []
    paths: List[str] = []
    query_embeds: List[torch.Tensor] = []
    gallery_embeds: List[torch.Tensor] = []

    query_mod = track_cfg["query_mod"]
    gallery_mod = track_cfg["gallery_mod"]

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding [{track_name}]"):
            ids.extend([str(x) for x in batch["gallery_id"]])
            paths.extend(batch["path"])

            if track_name == "cross":
                rgb = batch["rgb"].to(device)
                depth = batch["depth"].to(device)
                emb = model.encode(rgb, depth)
                query_embeds.append(emb)
                gallery_embeds.append(emb)
            else:
                # rgb or depth track
                if query_mod in batch:
                    xq = batch[query_mod]
                    if isinstance(xq, list):
                        xq = torch.stack(xq)
                    query_embeds.append(model(xq.to(device)))

                if gallery_mod in batch:
                    xg = batch[gallery_mod]
                    if isinstance(xg, list):
                        xg = torch.stack(xg)
                    gallery_embeds.append(model(xg.to(device)))

    return (
        ids,
        paths,
        torch.cat(query_embeds, dim=0),
        torch.cat(gallery_embeds, dim=0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────────────────────────────────────

def _build_rankings(
    ids: List[str],
    paths: List[str],
    query_mat: torch.Tensor,
    gallery_mat: torch.Tensor,
) -> List[Dict]:
    """Compute pairwise distances and produce ranked rows."""
    dists = torch.cdist(query_mat, gallery_mat).cpu()
    results: List[Dict] = []

    for i, qid in enumerate(ids):
        row = dists[i]
        sorted_idx = torch.argsort(row).tolist()
        rank = 1
        for g_idx in sorted_idx:
            gid = ids[g_idx]
            if gid == qid:
                continue
            results.append({
                "query_gallery_id": qid,
                "query_path": paths[i],
                "gallery_id": gid,
                "gallery_path": paths[g_idx],
                "rank": rank,
                "distance": float(row[g_idx]),
            })
            rank += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

_FIELDNAMES = [
    "query_gallery_id",
    "query_path",
    "gallery_id",
    "gallery_path",
    "rank",
    "distance",
]


def _save_csv(results: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved rankings → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-track evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_track(
    track_name: str,
    data_root: str,
    labels_csv: str,
    train_csv: str,
    batch_size: int,
    num_workers: int,
    eval_subdir: str | None,
    device: str,
) -> None:
    print(f"\n{'─'*60}")
    print(f" Evaluating track: {track_name.upper()}")
    print(f"{'─'*60}")

    track_cfg = TRACK_CONFIGS[track_name]

    # ── Dataset & loader ──────────────────────────────────────────────────────
    data_cfg = DataConfig(
        root=data_root,
        eval_csv=labels_csv,
        modality=track_cfg["data_modality"],
        val_mode="eval",
    )
    if eval_subdir is not None:
        data_cfg.eval_subdir = eval_subdir

    rgb_t, depth_t = build_transforms(data_cfg.transforms)
    dataset = UnifiedReIDDataset(
        csv_path=data_cfg.eval_csv,
        root=data_cfg.root,
        modality=data_cfg.modality,
        mode="eval",
        sequence=data_cfg.sequence,
        rgb_transform=rgb_t,
        depth_transform=depth_t,
        train_subdir=data_cfg.train_subdir,
        eval_subdir=data_cfg.eval_subdir,
        sampling_strategy=data_cfg.sequence.sampling,
        mask_rgb_with_depth=data_cfg.mask_rgb_with_depth,
        depth_mask_threshold=data_cfg.depth_mask_threshold,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = _load_model(track_cfg, train_csv, device)

    # ── Embeddings → rankings → CSV ───────────────────────────────────────────
    ids, paths, q_mat, g_mat = _extract_embeddings(
        loader, model, track_cfg, device, track_name
    )
    results = _build_rankings(ids, paths, q_mat, g_mat)
    _save_csv(results, track_cfg["output"])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate ranked gallery CSVs for ICPR-2026-TVRID tracks"
    )
    p.add_argument(
        "--track",
        choices=["rgb", "depth", "cross", "all"],
        default="all",
        help="Which track to evaluate (default: all)",
    )
    p.add_argument("--data_root",    default="data")
    p.add_argument("--labels_csv",   default="data/public_test_labels.csv")
    p.add_argument("--train_csv",    default="data/train_split.csv",
                   help="Needed only for depth track (num_classes)")
    p.add_argument("--eval_subdir",  default=None,
                   help="Override eval sub-directory (e.g. 'train' for local val)")
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("outputs", exist_ok=True)

    tracks_to_run = (
        list(TRACK_CONFIGS.keys()) if args.track == "all" else [args.track]
    )

    for track in tracks_to_run:
        evaluate_track(
            track_name=track,
            data_root=args.data_root,
            labels_csv=args.labels_csv,
            train_csv=args.train_csv,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            eval_subdir=args.eval_subdir,
            device=device,
        )

    print("\nAll tracks evaluated!")
