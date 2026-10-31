"""Dataset loading + caching.

MNIST is fetched once from OpenML and cached to data/mnist.npz (uint8 pixels,
int64 labels). Callers get float32 in [0, 1]. For S03 the first 60k training
images are also written as 1000-image shards (data/shards/shard_XXX.npz) so the
"load from disk" stage can be split across processes/batches.
"""
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MNIST_NPZ = DATA_DIR / "mnist.npz"
SHARD_DIR = DATA_DIR / "shards"
SHARD_SIZE = 1000
N_TRAIN = 60000


def _download_mnist():
    # python.org macOS builds ship without a CA bundle; use certifi's.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    from sklearn.datasets import fetch_openml
    t0 = time.time()
    print("[data] downloading mnist_784 from OpenML (one-off) ...", flush=True)
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True,
                        data_home=str(DATA_DIR / "openml_cache"), parser="liac-arff")
    X = np.asarray(X, dtype=np.uint8)
    y = np.asarray(y).astype(np.int64)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MNIST_NPZ, X=X, y=y)
    print(f"[data] cached {X.shape} to {MNIST_NPZ} in {time.time() - t0:.1f}s", flush=True)


def load_mnist_raw():
    """Return (X uint8 (70000, 784), y int64). Train = first 60k, test = last 10k."""
    if not MNIST_NPZ.exists():
        _download_mnist()
    with np.load(MNIST_NPZ) as d:
        return d["X"], d["y"]


def load_mnist():
    """Return (X_train, y_train, X_test, y_test), X float32 scaled to [0, 1]."""
    X, y = load_mnist_raw()
    X = X.astype(np.float32) / np.float32(255.0)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def make_blobs_data(n, n_features=64, centers=10, seed=0):
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=n, n_features=n_features, centers=centers,
                      cluster_std=2.0, random_state=seed)
    return np.ascontiguousarray(X, dtype=np.float32)


def ensure_shards():
    """Write the 60k MNIST training images as uint8 shards of 1000 images."""
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    n_shards = N_TRAIN // SHARD_SIZE
    if all((SHARD_DIR / f"shard_{i:03d}.npz").exists() for i in range(n_shards)):
        return n_shards
    X, _ = load_mnist_raw()
    for i in range(n_shards):
        np.savez(SHARD_DIR / f"shard_{i:03d}.npz", X=X[i * SHARD_SIZE:(i + 1) * SHARD_SIZE])
    return n_shards


def shard_path(i):
    return SHARD_DIR / f"shard_{i:03d}.npz"


def load_shard(i):
    with np.load(shard_path(i)) as d:
        return d["X"]


if __name__ == "__main__":
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    Xtr, ytr, Xte, yte = load_mnist()
    print(Xtr.shape, Xte.shape, Xtr.dtype, Xtr.min(), Xtr.max())
    print("shards:", ensure_shards())
