import os, argparse, math, time
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

from src.datamodules.bci2a import load_LOSO_pool, load_subject_dependent
from src.models.model import build_atcnet
from src.models.wrappers import build_ssl_projector
from src.selfsupervised.views import make_ssl_dataset
from src.selfsupervised.losses import nt_xent_loss

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
        return_ssl_feat=True,
    )

@tf.function
def train_step(ssl_model, v1, v2, optimizer, temperature: float):
    with tf.GradientTape() as tape:
        z1 = ssl_model(v1, training=True)
        z2 = ssl_model(v2, training=True)
        loss = nt_xent_loss(z1, z2, temperature=temperature)
    grads = tape.gradient(loss, ssl_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, ssl_model.trainable_variables))
    return loss

def run_loso(args):
    for tgt in range(1, args.n_sub + 1):
        fold_dir = os.path.join(args.results_dir, f"LOSO_{tgt:02d}")
        os.makedirs(fold_dir, exist_ok=True)

        (X_src, _), (X_tgt, y_tgt) = load_LOSO_pool(
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

        for ep in range(1, args.epochs + 1):
            losses = []
            for v1, v2 in ds:
                # expand to [B,1,C,T]
                v1 = tf.expand_dims(v1, 1)
                v2 = tf.expand_dims(v2, 1)
                losses.append(train_step(ssl_model, v1, v2, opt, args.temperature))
            if ep % args.log_every == 0:
                print(f"[LOSO {tgt:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

            if ep % args.save_every == 0 or ep == args.epochs:
                wpath = os.path.join(fold_dir, f"ssl_encoder_sub{tgt}_epoch{ep}.weights.h5")
                encoder.save_weights(wpath)

def run_subject_dependent(args):
    (X_tr, _), (X_te, _) = load_subject_dependent(
        args.data_root, args.subject,
        ea=args.ea, standardize=args.standardize,
        t1_sec=args.t1_sec, t2_sec=args.t2_sec
    )
    ds = make_ssl_dataset(
        X_tr, n_channels=args.n_channels, in_samples=args.in_samples,
        batch_size=args.batch_size, shuffle=True
    )

    sub_dir = os.path.join(args.results_dir, f"SUBJ_{args.subject:02d}")
    os.makedirs(sub_dir, exist_ok=True)

    encoder = build_encoder(args)
    ssl_model = build_ssl_projector(encoder, proj_dim=args.proj_dim, out_dim=args.out_dim)
    opt = tf.keras.optimizers.Adam(args.lr)

    for ep in range(1, args.epochs + 1):
        losses = []
        for v1, v2 in ds:
            v1 = tf.expand_dims(v1, 1)
            v2 = tf.expand_dims(v2, 1)
            losses.append(train_step(ssl_model, v1, v2, opt, args.temperature))
        if ep % args.log_every == 0:
            print(f"[SUBJ {args.subject:02d}] epoch {ep:03d}/{args.epochs}  ssl_loss={float(tf.reduce_mean(losses)): .4f}")

        if ep % args.save_every == 0 or ep == args.epochs:
            wpath = os.path.join(sub_dir, f"ssl_encoder_sub{args.subject}_epoch{ep}.weights.h5")
            encoder.save_weights(wpath)

def parse_args():
    p = argparse.ArgumentParser("ATCNet SSL pretraining")
    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="./results_ssl")
    p.add_argument("--n_sub", type=int, default=9)
    p.add_argument("--n_classes", type=int, default=4)
    p.add_argument("--n_channels", type=int, default=22)
    p.add_argument("--in_samples", type=int, default=1000)
    p.add_argument("--t1_sec", type=float, default=2.0)
    p.add_argument("--t2_sec", type=float, default=6.0)
    p.add_argument("--ea", action="store_true")
    p.add_argument("--no-ea", dest="ea", action="store_false")
    p.set_defaults(ea=True)
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--no-standardize", dest="standardize", action="store_false")
    p.set_defaults(standardize=True)
    p.add_argument("--per_block_standardize", action="store_true")
    p.add_argument("--no-per_block_standardize", dest="per_block_standardize", action="store_false")
    p.set_defaults(per_block_standardize=True)

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
    p.add_argument("--subject", type=int, default=1)

    # projector
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--out_dim", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    set_seed(args.seed)

    if args.loso:
        run_loso(args)
    else:
        run_subject_dependent(args)

if __name__ == "__main__":
    main()