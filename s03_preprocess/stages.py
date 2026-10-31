"""The four preprocessing stages, shared by all three variants.

(1) load uint8 shards from .npz, (2) normalise to [0,1] float32,
(3) standardise each image to zero mean / unit variance (per-sample, so the
stage is batch-independent and streamable), (4) HOG features per image.
"""
import numpy as np
from skimage.feature import hog

from common.data import load_shard

STAGES = ("load", "normalise", "standardise", "hog")
HOG_KW = dict(orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2),
              block_norm="L2-Hys", feature_vector=True)
N_FEAT = 324  # 3x3 blocks * 2x2 cells * 9 orientations for 28x28 input
EPS = np.float32(1e-8)


def load(shard_ids):
    return np.concatenate([load_shard(i) for i in shard_ids])


def normalise(raw, out=None):
    out = np.empty(raw.shape, np.float32) if out is None else out
    np.divide(raw, np.float32(255.0), out=out, dtype=np.float32)
    return out


def standardise(X):
    """In place, per image."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    X -= mu
    X /= (sd + EPS)
    return X


def hog_features(X, out=None):
    out = np.empty((X.shape[0], N_FEAT), np.float64) if out is None else out
    for i in range(X.shape[0]):
        out[i] = hog(X[i].reshape(28, 28), **HOG_KW)
    return out
