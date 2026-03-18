# ICPR-2026-TVRID

**Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera (TVRID)**

---

## Project Structure

```
ICPR-2026-TVRID/
│
├── config/
│   ├── data/
│   │   └── default.yaml          # Data paths, transforms, sequence config
│   └── train.yaml                # Training hyper-parameters for all 3 tracks
│
├── data/
│
├── utils/                        # ── Shared across all tracks ──
│   ├── __init__.py
│   ├── data.py                   # DataConfig, Dataset, DataModule, Transforms
│   └── models.py                 # ConvNeXtRGBEncoder, ConvNeXtDepthEncoder, TripletLoss
│
├── models/                       # ── Track-specific Lightning modules ──
│   ├── __init__.py
│   ├── rgb_model.py              # Track 1 – RGBReIDLightning
│   ├── depth_model.py            # Track 2 – DepthOnlyTrainer + DepthPreprocessor
│   └── cross_model.py            # Track 3 – FusionReID + FusionEncoder
│
├── prepare_data.py               # Create train/val identity-disjoint splits
├── train_rgb.py                  # Train Track 1 (RGB)
├── train_depth.py                # Train Track 2 (Depth)
├── train_cross.py                # Train Track 3 (Cross-modal Fusion)
├── eval_generate.py              # Generate ranked gallery CSVs for all tracks
├── eval_score.py                 # Compute mAP / CMC from ranking CSVs
└── requirements.txt
```

---

## Tracks

| Track | Modality | Query | Gallery | Model |
|-------|----------|-------|---------|-------|
| 1 – RGB | RGB only | RGB | RGB | `RGBReIDLightning` |
| 2 – Depth | Depth only | Depth | Depth | `DepthOnlyTrainer` |
| 3 – Cross | RGB-D Fusion | RGB | Depth | `FusionReID` |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data splits
```bash
python prepare_data.py
# Outputs: data/train_split.csv, data/valid_split.csv
```

### 3. Train

```bash
# Track 1 – RGB
python train_rgb.py

# Track 2 – Depth
python train_depth.py

# Track 3 – Cross-modal Fusion
python train_cross.py
```

All scripts accept CLI flags to override defaults:
```bash
python train_rgb.py --epochs 30 --batch_size 8 --lr 1e-4
python train_depth.py --epochs 40 --margin 0.5
python train_cross.py --epochs 25 --embedding_size 512
```

### 4. Generate rankings
```bash
python eval_generate.py --track all
# or for a single track:
python eval_generate.py --track rgb
```

### 5. Score
```bash
python eval_score.py --track all
```

---

## Architecture

### Shared Encoders (`utils/models.py`)

Both RGB and Depth encoders use **ConvNeXt-Tiny** as backbone:

- `ConvNeXtRGBEncoder` – standard 3-channel input, ImageNet pretrained.
- `ConvNeXtDepthEncoder` – first conv replaced with 1-channel input, remaining weights transferred from pretrained model.

Both output **L2-normalised** embeddings of configurable dimension (default 256).

### Loss Function

All three tracks use **TripletLoss** (squared-Euclidean, soft margin).  
RGB and Cross tracks additionally use **CrossEntropyLoss** on a BNNeck projection head.  
Depth track adds CE with a 0.5 weighting coefficient to stabilise training.

### Track 3 – Fusion

`FusionEncoder` concatenates RGB and Depth embeddings then projects through a single linear + BN + ReLU layer to produce the final cross-modal embedding.

---
