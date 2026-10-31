"""Sequential brute-force k-NN (no Pool) + kernels shared with the parallel versions.

dist^2(q, t) = ||q||^2 + ||t||^2 - 2 q.t, computed per (query block x train chunk)
GEMM. Query blocks (QBLOCK rows) and train chunks (TCHUNK rows) are fixed and
globally aligned, so every strategy executes byte-identical GEMMs.

Neighbour selection is the k smallest (distance, train index) pairs in
lexicographic order. This makes ties deterministic and makes "local top-k per
train partition, then merge" give exactly the global top-k.
"""
import time

import numpy as np

QBLOCK = 125
TCHUNK = 1250
K = 5
N_CLASSES = 10


def n_chunks(n):
    return (n + TCHUNK - 1) // TCHUNK


def block_distances(Qb, qnorm, X, xnorm, c0, c1):
    """Distances from query block Qb to train chunks [c0, c1)."""
    s, e = c0 * TCHUNK, min(c1 * TCHUNK, X.shape[0])
    D = np.empty((Qb.shape[0], e - s), dtype=np.float32)
    for c in range(c0, c1):
        a, b = c * TCHUNK, min((c + 1) * TCHUNK, X.shape[0])
        D[:, a - s:b - s] = qnorm[:, None] + xnorm[None, a:b] - 2.0 * (Qb @ X[a:b].T)
    return D


def _lex_select(d, i, k):
    """Per row, the k smallest (d, i) pairs in lexicographic order (inputs: rows x m)."""
    o = np.argsort(i, axis=1, kind="stable")
    d, i = np.take_along_axis(d, o, 1), np.take_along_axis(i, o, 1)
    o = np.argsort(d, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(d, o, 1), np.take_along_axis(i, o, 1)


def topk_lex(D, k, offset=0):
    """k smallest (distance, global index) per row of D; global index = column + offset."""
    rows = D.shape[0]
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    pd = np.take_along_axis(D, part, 1)
    kth = pd.max(axis=1)
    out_d, out_i = _lex_select(pd, part.astype(np.int64), k)
    ties = np.flatnonzero((D <= kth[:, None]).sum(axis=1) > k)  # rows with ties at the k-th distance
    for r in ties:
        cand = np.flatnonzero(D[r] <= kth[r])
        dd, ii = _lex_select(D[r, cand][None, :], cand[None, :].astype(np.int64), k)
        out_d[r], out_i[r] = dd[0], ii[0]
    return out_d, out_i + offset


def merge_topk(ds, idxs, k):
    """Merge per-partition top-k lists (lists of rows x k arrays) into the global top-k."""
    return _lex_select(np.concatenate(ds, axis=1), np.concatenate(idxs, axis=1), k)


def vote(nn_idx, y):
    votes = y[nn_idx]
    return np.array([np.bincount(v, minlength=N_CLASSES).argmax() for v in votes], dtype=np.int64)


def predict_blocks(Q, X, xnorm, y, b0, b1, k=K):
    """Query-partitioned kernel: predict labels for query blocks [b0, b1) against all of X."""
    nc = n_chunks(X.shape[0])
    preds = []
    for b in range(b0, b1):
        Qb = Q[b * QBLOCK:min((b + 1) * QBLOCK, Q.shape[0])]
        qnorm = np.einsum("ij,ij->i", Qb, Qb)
        _, nn = topk_lex(block_distances(Qb, qnorm, X, xnorm, 0, nc), k)
        preds.append(vote(nn, y))
    return np.concatenate(preds) if preds else np.empty(0, dtype=np.int64)


def topk_train_part(Q, X, xnorm, c0, c1, k=K):
    """Train-partitioned kernel: local top-k of ALL queries against train chunks [c0, c1)."""
    ds, idxs = [], []
    for b in range(0, (Q.shape[0] + QBLOCK - 1) // QBLOCK):
        Qb = Q[b * QBLOCK:min((b + 1) * QBLOCK, Q.shape[0])]
        qnorm = np.einsum("ij,ij->i", Qb, Qb)
        d, i = topk_lex(block_distances(Qb, qnorm, X, xnorm, c0, c1), k, offset=c0 * TCHUNK)
        ds.append(d)
        idxs.append(i)
    return np.concatenate(ds), np.concatenate(idxs)


def knn_sequential(X, y, Q):
    t0 = time.perf_counter()
    xnorm = np.einsum("ij,ij->i", X, X)
    nblocks = (Q.shape[0] + QBLOCK - 1) // QBLOCK
    preds = predict_blocks(Q, X, xnorm, y, 0, nblocks)
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0, "result": preds}


def knn_sequential_chunked(X, y, Q, n_parts=8):
    """Diagnostic: the 8-worker query-partitioned decomposition executed in-process."""
    from common.harness import split_ranges
    t0 = time.perf_counter()
    xnorm = np.einsum("ij,ij->i", X, X)
    nblocks = (Q.shape[0] + QBLOCK - 1) // QBLOCK
    preds = np.concatenate([predict_blocks(Q, X, xnorm, y, b0, b1)
                            for b0, b1 in split_ranges(nblocks, n_parts)])
    t1 = time.perf_counter()
    return {"exec": t1 - t0, "comm": 0.0, "sync": 0.0, "sched": 0.0, "result": preds}
