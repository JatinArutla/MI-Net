# MI-Net

Motor-imagery EEG classification on **BCI Competition IV-2a (BNCI 2014-001)** using **ATCNet**, with optional **self-supervised pretraining (NT-Xent)** and supervised finetuning.

---

## Task

- **Supervised modes**
  - **LOSO** (Leave-One-Subject-Out): pooled sources → train/val split
  - **Subject-dependent** (per-subject loop): A0sT → train/val split
- **Optional** warm-start from SSL encoder weights.

**Dataset & windowing:** 9 subjects, 22 channels, 4 classes, 250 Hz. Trials use **[2.0 s, 6.0 s]** after cue.

---

## Why these changes

**ATCNet-M** = ATCNet with minor architectural hyperparameter changes:
- Wider stem (**F1 = 16 vs 4**)
- Reduced downsampling (**pool7 vs 8**)
- **ELU** activations (vs ReLU)
- Removal of **L2/max-norm** on the skip 1×1 conv

All other blocks and ordering match the original ATCNet.

---

## Standardization options

- **Per-channel z-score:** μ/σ per channel over (N,T)[,(M)] from train; one scale/shift across time (preserves temporal dynamics).
- **Per-timepoint standardization:** μ/σ per channel & time index across trials; reweights the timeline and can distort amplitude/latency patterns; fit per fold to avoid leakage.

---

## Results (reported)

**Per-subject average test accuracy (%)**

| Network                                       | Acc   |
|-----------------------------------------------|-------|
| ATCNet + per-timepoint standardization        | 75.69 |
| ATCNet + EA + per-timepoint standardization   | 77.04 |
| ATCNet-M + EA + per-timepoint standardization | 77.39 |
| ATCNet-M + EA + per-channel z-score           | 79.17 |

**LOSO supervised vs. LOSO supervised + SSL init**

| Sub | Acc  |   | Sub | Acc  |
|-----|------|---|-----|------|
| 01  | 77.08|   | 01  | 77.26|
| 02  | 50.35|   | 02  | 49.31|
| 03  | 81.77|   | 03  | 84.38|
| 04  | 61.11|   | 04  | 59.20|
| 05  | 55.21|   | 05  | 56.08|
| 06  | 59.38|   | 06  | 59.03|
| 07  | 70.49|   | 07  | 72.05|
| 08  | 75.17|   | 08  | 80.38|
| 09  | 73.09|   | 09  | 71.35|
| **Avg** | **67.07** | | **Avg** | **67.67** |

---

## Setup

```python
DATA_ROOT = "../four-class-motor-imagery-bnci-001-2014"

# If running in a notebook:
import sys, os
sys.path.insert(0, os.path.abspath("src"))

## Usage

### Self-supervised pretraining (`train_ssl.py`)

**1) LOSO (loop over targets 1..9)**  
[Trains on pooled sources for each target; saves per-fold encoder weights.]
python train_ssl.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_ssl \
  --loso \
  --epochs 100 --batch_size 256 --lr 1e-3 \
  --probe_every 25 --probe_on target

**2) Subject-dependent (loop over all subjects)
[Omit --subject to train SSL for subjects 1..9.]
python train_ssl.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_ssl_subj \
  --no-loso \
  --epochs 100 --batch_size 256 --lr 1e-3

### Supervised finetuning (finetune.py)

Common flags
--loso (default on): pooled sources → train/val split
--no-loso: subject-dependent (A0sT) → train/val split
--ssl_weights: optional path template to warm-start from SSL encoder weights
LOSO: ./results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5
Subject-dependent: ./results_ssl_subj/SUBJ_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5

**1) LOSO supervised (from scratch)
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_loso \
  --loso \
  --epochs 500 --batch_size 64 --lr 1e-3

**2)LOSO supervised with SSL weights
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_loso_ssl \
  --loso \
  --ssl_weights "./results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5" \
  --epochs 500 --batch_size 64 --lr 1e-3

**3)Subject-dependent supervised (loop all subjects)
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_subj \
  --no-loso \
  --epochs 500 --batch_size 64 --lr 1e-3

**4)Subject-dependent supervised with SSL weights
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_subj_ssl \
  --no-loso \
  --ssl_weights "./results_ssl_subj/SUBJ_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5" \
  --epochs 500 --batch_size 64 --lr 1e-3