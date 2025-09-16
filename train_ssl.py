# train_ssl.py
# Self-supervised pretraining for ATCNet on BCI IV-2a
# - Supports LOSO and subject-dependent (single subject or loop all)
# - Optional linear / k-NN probes during training

import os, argparse
os.environ["TF_DISABLE_LAYOUT_OPTIMIZER"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import tensorflow as tf

tf.keras.backend.set_image_data_format("channels_last")
tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.models.model import build_atcnet
from src.models.wrappers import build_ssl_projector
from src.selfsupervised.views import make_ssl_dataset
from src.selfsupervised.losses import nt_xent_loss


# ----------------- Utils -----------------

def set_seed(seed: int = 1):
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def build_encoder(args):
    return build_atcnet(
        n_classes=args.n_classes,
        in_chans=args.n_channels,
        in_samples=args.in_samples,
        n_windows=args.n_windows,
        attention=args.attention,
        eegn_F1=args.eegn_F1,
        eegn_D=args.eegn_D,
        eegn_kernel=args.eegn_kernel,
        eegn_pool=args.eegn_pool,
        eegn_dropout=args.eegn_dropout,
        tcn_depth=args.tcn_depth,
        tcn_kernel=args.tcn_kernel,
        tcn_filters=args.tcn_filters,
        tcn_dropout=args.tcn_dropout,
        tcn_activation=args.tcn_activation,
        fuse=args.fuse,
        from_logits=False,
        return_ssl_feat=True,  # exposes averaged per-window feature as second output
    )

def _pack_X(X):  # [N,C,T] -> [N,1,C,T]
    X = X.astype(np.float32, copy=False)
    return X if X.ndim == 4 else X[:, None, ...]

def _to_b1ct(x):
    # If dataset yields [B,C,T] → make [B,1,C,T]; if already [B,1,C,T] → pass-through
    return x if x.shape.rank == 4 else tf.expand_dims(x, 1)

def _probe_now(encoder, X, y, split: float, k: int):
    feat_model = tf.keras.Model(encoder.input, encoder.outputs[1], name="feature_tap")
    Z = feat_model(_pack_X(X), training=False).numpy()
    Ztr, Zva, ytr, yva = train_test_split(Z, y, test_size=split, random_state=42, stratify=y)
    lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
    acc_lr = accuracy_score(yva, lr.predict(Zva))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1).fit(Ztr, ytr)
    acc_knn = accuracy_score(yva, knn.predict(Zva))
    return acc_lr, acc_knn


# ----------------- Runners -----------------

def run_loso(args):
    for tgt in range(1, args.n_sub + 1):
        fold_dir = os.path.join(args.results_dir, f"LOSO_{tgt:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        (X_src, y_src), (X_tgt, y_tgt) = load_LOSO_pool(
            args.data_root, tgt,
            n_sub=args.n_sub, ea=args.ea, standardize=args.standardize,
            per_block_standardize=args.per_block_standardize,
            t1_sec=args.t1_sec, t2_sec=args.t2_sec
        )

        ds = make_ssl_dataset(
            X_src, n_channels=args.n_channels, in_samples=args.in_samples,
            batch_size=args.batch_size, shuffle=True
        )

        encoder = build_encoder(args)
        ssl_model = build_ssl_projector(encoder, proj_dim=args.proj_dim, out_dim=args.out_dim)
        opt = tf.keras.optimizers.Adam(args.lr)
        temperature = tf.constant(args.temperature, tf.float32)

        @tf.function(reduce_retracing=True)  # comment out to run eagerly
        def train_step(v1, v2):
            with tf.GradientTape() as tape:
                z1 = ssl_model(v1, training=True)
                z2 = ssl_model(v2, training=True)
                loss = nt_xent_loss(z1, z2, temperature)
            grads = tape.gradient(loss, ssl_model.trainable_variables)
            grads_vars = [(g, v) for g, v in zip(grads, ssl_model.trainable_variables) if g is not None]
            opt.apply_gradients(grads_vars)
            return loss

        # warm-up call to build variables once
        warm = next(iter(ds))
        _ = train_step(_to_b1ct(warm[0]), _to_b1ct(warm[1]))

        for ep in range(1, args.epochs + 1):
            losses = []
            for v1, v2 in ds:
                losses.append(train_step(_to_b1ct(v1), _to_b1ct(v2)))
            if ep % args.log_every == 0:
                print(f"[LOSO {tgt:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

            if args.probe_every > 0 and ep % args.probe_every == 0:
                if args.probe_on == "target":
                    acc_lr, acc_knn = _probe_now(encoder, X_tgt, y_tgt, args.probe_split, args.probe_k)
                else:
                    acc_lr, acc_knn = _probe_now(encoder, X_src, y_src, args.probe_split, args.probe_k)
                print(f"   ↳ probe@{ep}: linear={acc_lr:.3f}  knn@{args.probe_k}={acc_knn:.3f}")

            if ep % args.save_every == 0 or ep == args.epochs:
                wpath = os.path.join(fold_dir, f"ssl_encoder_sub{tgt}_epoch{ep}.weights.h5")
                encoder.save_weights(wpath)


def run_subject_dependent_one(args, sub: int):
    (X_tr, y_tr), (X_te, y_te) = load_subject_dependent(
        args.data_root, sub,
        ea=args.ea, standardize=args.standardize,
        t1_sec=args.t1_sec, t2_sec=args.t2_sec
    )
    ds = make_ssl_dataset(
        X_tr, n_channels=args.n_channels, in_samples=args.in_samples,
        batch_size=args.batch_size, shuffle=True
    )

    sub_dir = os.path.join(args.results_dir, f"SUBJ_{sub:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    encoder = build_encoder(args)
    ssl_model = build_ssl_projector(encoder, proj_dim=args.proj_dim, out_dim=args.out_dim)
    opt = tf.keras.optimizers.Adam(args.lr)
    temperature = tf.constant(args.temperature, tf.float32)

    @tf.function(reduce_retracing=True)
    def train_step(v1, v2):
        with tf.GradientTape() as tape:
            z1 = ssl_model(v1, training=True)
            z2 = ssl_model(v2, training=True)
            loss = nt_xent_loss(z1, z2, temperature)
        grads = tape.gradient(loss, ssl_model.trainable_variables)
        grads_vars = [(g, v) for g, v in zip(grads, ssl_model.trainable_variables) if g is not None]
        opt.apply_gradients(grads_vars)
        return loss

    warm = next(iter(ds))
    _ = train_step(_to_b1ct(warm[0]), _to_b1ct(warm[1]))

    for ep in range(1, args.epochs + 1):
        losses = []
        for v1, v2 in ds:
            losses.append(train_step(_to_b1ct(v1), _to_b1ct(v2)))
        if ep % args.log_every == 0:
            print(f"[SUBJ {sub:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

        if args.probe_every > 0 and ep % args.probe_every == 0:
            Xp, yp = (X_te, y_te) if args.probe_on == "target" else (X_tr, y_tr)
            acc_lr, acc_knn = _probe_now(encoder, Xp, yp, args.probe_split, args.probe_k)
            print(f"   ↳ probe@{ep}: linear={acc_lr:.3f}  knn@{args.probe_k}={acc_knn:.3f}")

        if ep % args.save_every == 0 or ep == args.epochs:
            wpath = os.path.join(sub_dir, f"ssl_encoder_sub{sub}_epoch{ep}.weights.h5")
            encoder.save_weights(wpath)


def run_subject_dependent_all(args):
    for sub in range(1, args.n_sub + 1):
        run_subject_dependent_one(args, sub)


# ----------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser("ATCNet SSL pretraining (+ optional linear/kNN probe)")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results_ssl")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)
    p.add_argument("--ea", action="store_true"); p.add_argument("--no-ea", dest="ea", action="store_false"); p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true"); p.add_argument("--no-standardize", dest="standardize", action="store_false"); p.set_defaults(standardize=True)
    p.add_argument("--per_block_standardize", action="store_true"); p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false"); p.set_defaults(per_block_standardize=True)

    # model
    p.add_argument("--n_windows", type=int, default=5)
    p.add_argument("--attention", type=str, default="mha", choices=["mha", "mhla", "none", ""])
    p.add_argument("--eegn_F1", type=int, default=16)
    p.add_argument("--eegn_D", type=int, default=2)
    p.add_argument("--eegn_kernel", type=int, default=64)
    p.add_argument("--eegn_pool", type=int, default=7)
    p.add_argument("--eegn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_depth", type=int, default=2)
    p.add_argument("--tcn_kernel", type=int, default=4)
    p.add_argument("--tcn_filters", type=int, default=32)
    p.add_argument("--tcn_dropout", type=float, default=0.3)
    p.add_argument("--tcn_activation", type=str, default="elu")
    p.add_argument("--fuse", type=str, default="average", choices=["average", "concat"])

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)

    # mode
    p.add_argument("--loso", action="store_true")
    p.add_argument("--subject", type=int, default=None,
                   help="Subject ID for subject-dependent mode; omit to loop 1..n_sub")

    # projector
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=64)

    # probes
    p.add_argument("--probe_every", type=int, default=0, help="run linear/kNN probe every N epochs (0=off)")
    p.add_argument("--probe_split", type=float, default=0.2, help="validation split for probe")
    p.add_argument("--probe_k", type=int, default=5, help="k for k-NN probe")
    p.add_argument("--probe_on", type=str, default="target", choices=["target","source"], help="which labeled set to probe on")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    if args.loso:
        run_loso(args)
    else:
        if args.subject is None:
            run_subject_dependent_all(args)
        else:
            run_subject_dependent_one(args, args.subject)


if __name__ == "__main__":
    main()