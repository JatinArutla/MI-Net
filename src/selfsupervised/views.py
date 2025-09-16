import numpy as np
import tensorflow as tf

# ----- NumPy augs on [C,T] -----
def _amp_add(x, low=1.0, high=4.0):
    a = np.random.uniform(low, high); return x + a

def _amp_scale(x, low=2.0, high=4.0):
    s = np.random.uniform(low, high); return x * s

def _time_warp(x, seg_min=4, seg_max=10, scale=0.5):
    C, T = x.shape; m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int); parts = []
    for i in range(m):
        seg = x[:, cuts[i]:cuts[i+1]]
        f = np.random.uniform(1.0 - scale, 1.0 + scale)
        L = max(1, int(seg.shape[1] * f))
        grid = np.linspace(0, seg.shape[1] - 1, L)
        parts.append(np.stack([np.interp(grid, np.arange(seg.shape[1]), seg[c]) for c in range(C)], 0))
    warp = np.concatenate(parts, 1)
    gridT = np.linspace(0, warp.shape[1] - 1, T)
    return np.stack([np.interp(gridT, np.arange(warp.shape[1]), warp[c]) for c in range(C)], 0)

def _cutout_resize(x, seg_min=4, seg_max=10):
    C, T = x.shape; m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    p = np.random.randint(0, m)
    keep = [x[:, cuts[i]:cuts[i+1]] for i in range(m) if i != p]
    y = np.concatenate(keep, 1)
    grid = np.linspace(0, y.shape[1] - 1, T)
    return np.stack([np.interp(grid, np.arange(y.shape[1]), y[c]) for c in range(C)], 0)

def _cutout_zero(x, seg_min=4, seg_max=10):
    C, T = x.shape; m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    p = np.random.randint(0, m); y = x.copy()
    y[:, cuts[p]:cuts[p+1]] = 0.0; return y

def _crop_resize(x, ratio_low=0.4, ratio_high=0.8):
    C, T = x.shape; r = np.random.uniform(ratio_low, ratio_high)
    L = max(1, int(T * r)); st = np.random.randint(0, T - L + 1)
    crop = x[:, st:st + L]
    grid = np.linspace(0, L - 1, T)
    return np.stack([np.interp(grid, np.arange(L), crop[c]) for c in range(C)], 0)

def _flip_time(x): return x[:, ::-1]

def _permute(x, seg_min=4, seg_max=10):
    C, T = x.shape; m = np.random.randint(seg_min, seg_max + 1)
    cuts = np.linspace(0, T, m + 1, dtype=int)
    segs = [x[:, cuts[i]:cuts[i+1]] for i in range(m)]
    np.random.shuffle(segs)
    return np.concatenate(segs, 1)

_AUGS = [_amp_add, _amp_scale, _time_warp, _cutout_resize, _cutout_zero, _crop_resize, _flip_time, _permute]

def two_random_augs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ops = np.random.choice(_AUGS, size=2, replace=False)
    v1 = ops[0](x.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    v2 = ops[1](x.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    return v1, v2

# ----- tf.data builder -----
def _two_views_np(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return two_random_augs(x)

def make_ssl_dataset(
    X: np.ndarray,
    *,
    n_channels: int,
    in_samples: int,
    batch_size: int = 256,
    shuffle: bool = True,
) -> tf.data.Dataset:
    N, C, T = X.shape
    assert C == n_channels and T == in_samples

    def gen():
        idx = np.arange(N)
        if shuffle: np.random.shuffle(idx)
        for i in idx:
            yield X[i].astype(np.float32, copy=False)

    ds = tf.data.Dataset.from_generator(
        gen, output_signature=tf.TensorSpec(shape=(n_channels, in_samples), dtype=tf.float32)
    )
    ds = ds.map(lambda x: tf.numpy_function(_two_views_np, [x], Tout=(tf.float32, tf.float32)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda v1, v2: (
            tf.ensure_shape(v1, (n_channels, in_samples)),
            tf.ensure_shape(v2, (n_channels, in_samples))
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(lambda v1, v2: (tf.expand_dims(v1, 0), tf.expand_dims(v2, 0)),  # -> [1,C,T]
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds