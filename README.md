# MI-Net

ATCNet + Self-Supervised Pretraining for BCI IV-2a

Motor imagery EEG classification using ATCNet with:

Self-supervised pretraining (NT-Xent) on BCI Competition IV-2a

Supervised finetuning in two modes:

LOSO (Leave-One-Subject-Out): pooled sources → validation split

Subject-dependent (per-subject loop): A0sT → validation split

Optional warm-start from SSL encoder weights.

Dataset: BNCI 2014-001 (BCI IV-2a): 9 subjects, 22 channels, 4 classes, 250 Hz.
Trial window used in these scripts: [2.0s, 6.0s] after cue.

Results
Test accuracy (%) — LOSO supervised
Sub	Acc
01	77.08
02	50.35
03	81.77
04	61.11
05	55.21
06	59.38
07	70.49
08	75.17
09	73.09
Avg	 67.07
Test accuracy (%) — LOSO supervised + SSL initialization
Sub	Acc
01	77.26
02	49.31
03	84.38
04	59.20
05	56.08
06	59.03
07	72.05
08	80.38
09	71.35
Avg	67.67

DATA_ROOT = "../four-class-motor-imagery-bnci-001-2014"

If you run in a notebook, add the repo to PYTHONPATH:
import sys, os
sys.path.insert(0, os.path.abspath("src"))

Self-supervised pretraining
1) LOSO (loop over targets 1..9)
Trains on pooled sources for each target; saves per-fold encoder weights.

python train_ssl.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_ssl \
  --loso \
  --epochs 100 --batch_size 256 --lr 1e-3 \
  --probe_every 25 --probe_on target

2) Subject-dependent (loop over all subjects)
Omit --subject to train SSL for all subjects 1..9:

python train_ssl.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_ssl_subj \
  --no-loso \
  --epochs 100 --batch_size 256 --lr 1e-3

Supervised finetuning

Script: finetune.py.
It trains and reports validation metrics; models are saved per subject and run.

Common flags:

--loso (default on): pooled sources → train/val split (held-out target is not used here).

--no-loso: subject-dependent using A0sT with a train/val split.

--ssl_weights: optional path template to warm-start from SSL encoder weights:

LOSO: ./results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5

Subject-dependent: ./results_ssl_subj/SUBJ_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5

A) LOSO supervised (from scratch)
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_loso \
  --loso \
  --epochs 500 --batch_size 64 --lr 1e-3

B) LOSO supervised with SSL weights
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_loso_ssl \
  --loso \
  --ssl_weights "./results_ssl/LOSO_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5" \
  --epochs 500 --batch_size 64 --lr 1e-3

C) Subject-dependent supervised (loop all subjects)
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_subj \
  --no-loso \
  --epochs 500 --batch_size 64 --lr 1e-3

D) Subject-dependent supervised with SSL weights
python finetune.py \
  --data_root "/kaggle/input/four-class-motor-imagery-bnci-001-2014" \
  --results_dir ./results_sup_subj_ssl \
  --no-loso \
  --ssl_weights "./results_ssl_subj/SUBJ_{sub:02d}/ssl_encoder_sub{sub}_epoch100.weights.h5" \
  --epochs 500 --batch_size 64 --lr 1e-3