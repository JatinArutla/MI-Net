import numpy as np

def _eigh_inv_sqrt(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, v = np.linalg.eigh(M + eps * np.eye(M.shape[0], dtype=M.dtype))
    return v @ np.diag(1.0 / np.sqrt(w)) @ v.T

def ea_align_trials(X: np.ndarray) -> np.ndarray:
    """EA for one subject/session block. X: [N,C,T] -> [N,C,T]."""
    covs = [x @ x.T / x.shape[1] for x in X]
    Rbar = np.mean(covs, axis=0)
    W = _eigh_inv_sqrt(Rbar.astype(np.float64)).astype(np.float32)
    return np.asarray([W @ x for x in X], dtype=np.float32)

def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Channel-wise z-score stats. 3D: mean/std over (N,T). 4D: over (N,M,T)."""
    Xf = X.astype(np.float32, copy=False)
    if Xf.ndim == 3:
        mu = Xf.mean(axis=(0, 2), keepdims=True)
        sd = Xf.std(axis=(0, 2), keepdims=True) + 1e-8
    elif Xf.ndim == 4:
        mu = Xf.mean(axis=(0, 1, 3), keepdims=True)
        sd = Xf.std(axis=(0, 1, 3), keepdims=True) + 1e-8
    else:
        raise ValueError(f"Expected 3D/4D array, got {Xf.ndim}D.")
    return mu.astype(np.float32), sd.astype(np.float32)

def apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32, copy=False) - mu) / sd).astype(np.float32)

def standardize_pair(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if X_train.ndim != X_test.ndim:
        raise ValueError(f"X_train.ndim ({X_train.ndim}) != X_test.ndim ({X_test.ndim})")
    mu, sd = fit_standardizer(X_train)
    return apply_standardizer(X_train, mu, sd), apply_standardizer(X_test, mu, sd)

def standardize_loso_block(X: np.ndarray) -> np.ndarray:
    """Within-block standardization used when pooling subjects (LOSO helper)."""
    mu, sd = fit_standardizer(X)
    return apply_standardizer(X, mu, sd)