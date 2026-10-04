"""Expected-metric grid decoding: maximise E[grid_score] under the per-cell posterior."""
import numpy as np


def _scores(pred, S):
    """pred: (16,) int; S: (K,16) int -> (K,) grid scores."""
    K = S.shape[0]
    num = np.zeros(K); den = np.zeros(K)
    for v in (1, 2, 3):
        pv = (pred == v)
        gv = (S == v)
        inter = (pv[None, :] & gv).sum(1)
        uni = (pv[None, :] | gv).sum(1)
        m = uni > 0
        num[m] += inter[m] / uni[m]
        den[m] += 1
    macro = np.where(den > 0, num / np.maximum(den, 1), 1.0)
    pa = pred > 0
    ga = S > 0
    inter = (pa[None, :] & ga).sum(1)
    tot = pa.sum() + ga.sum(1)
    f1 = np.where(tot > 0, 2 * inter / np.maximum(tot, 1), 1.0)
    return 0.70 * macro + 0.30 * f1


def decode(prob, K=256, passes=3, seed=0):
    """prob: (4,4,4) class-first probabilities -> (16,) predicted grid."""
    p = prob.reshape(4, 16).T                      # 16 x 4
    p = p / p.sum(1, keepdims=True)
    rng = np.random.default_rng(seed)
    cum = p.cumsum(1)
    u = rng.random((K, 16, 1))
    S = (u > cum[None, :, :]).sum(2).astype(np.int8)
    pred = p.argmax(1).astype(np.int8)
    best = _scores(pred, S).mean()
    for _ in range(passes):
        changed = False
        for i in range(16):
            cur = pred[i]
            for v in (0, 1, 2, 3):
                if v == cur:
                    continue
                pred[i] = v
                s = _scores(pred, S).mean()
                if s > best + 1e-9:
                    best = s; cur = v; changed = True
                else:
                    pred[i] = cur
        if not changed:
            break
    return pred.astype(int), best
