"""Decode the grid as a stack of ordered row bands instead of 16 independent cells.

Measured on the training golds: the active rows are contiguous in 100% of receipts, each
class occupies a contiguous band of rows, and the bands run 1 -> 2 -> 3 down the page
(99.4% / 100%).  Cells inside a band may still be blank, which is why this is a band search
rather than a row labelling.  Restricting the argmax to structurally possible maps removes
exactly the failures macro-IoU punishes hardest: a stray class-2 cell in the middle of the
menu, or a class the receipt does not have at all.
"""
import json, itertools, sys
import numpy as np, pandas as pd
from scorer import grid_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'

BANDS = [None] + [(a, b) for a in range(4) for b in range(a, 4)]


def combos():
    """Every structurally legal (band1, band2, band3); class 3 is always present."""
    out = []
    for i1, b1 in enumerate(BANDS):
        for i2, b2 in enumerate(BANDS):
            for i3, b3 in enumerate(BANDS):
                if b3 is None:
                    continue
                if b1 and b2 and b1[1] > b2[1]:
                    continue
                if b2 and b2[1] > b3[1]:
                    continue
                if b1 and not b2 and b1[1] > b3[1]:
                    continue
                rows = set()
                for b in (b1, b2, b3):
                    if b:
                        rows |= set(range(b[0], b[1] + 1))
                r = sorted(rows)
                if r[-1] - r[0] + 1 != len(r):        # active rows must be contiguous
                    continue
                out.append((b1, b2, b3))
    return out


COMBOS = combos()
# allowed[k] : (16,4) boolean mask of classes permitted per cell under combo k
ALLOW = np.zeros((len(COMBOS), 16, 4), bool)
for k, (b1, b2, b3) in enumerate(COMBOS):
    for cell in range(16):
        r = cell // 4
        ALLOW[k, cell, 0] = True
        for v, b in ((1, b1), (2, b2), (3, b3)):
            if b and b[0] <= r <= b[1]:
                ALLOW[k, cell, v] = True


def decode(P4, bias=(0., 0., 0., 0.), pres=None, lam=0.0):
    """P4 (n,4,4,4) class probabilities -> (n,16) labels."""
    L = np.log(np.clip(P4.reshape(len(P4), 4, 16), 1e-9, None)) + np.asarray(bias, np.float32)[None, :, None]
    L = L.transpose(0, 2, 1)                                   # n,16,4
    NEG = -1e9
    masked = np.where(ALLOW[None], L[:, None], NEG)            # n,K,16,4
    bestc = masked.max(3)                                      # n,K,16
    tot = bestc.sum(2)                                         # n,K
    if pres is not None and lam:
        has = np.array([[1.0 * (b is not None) for b in c] for c in COMBOS], np.float32)  # K,3
        lp = np.log(np.clip(pres[:, 1:], 1e-6, 1 - 1e-6))
        ln = np.log(np.clip(1 - pres[:, 1:], 1e-6, 1 - 1e-6))
        tot = tot + lam * (lp @ has.T + ln @ (1 - has).T)
    k = tot.argmax(1)
    return masked[np.arange(len(P4)), k].argmax(2)


def main():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    G = np.array([a['semantic_grid'] for a in A])
    n = len(G)
    og = np.load(W + r'\cache\oofC_g.npy')
    print(f'{len(COMBOS)} legal band configurations', flush=True)

    def sc(p):
        return float(np.mean([grid_score(p[i], G[i]) for i in range(n)]))

    base = sc(og.reshape(n, 4, 16).argmax(1))
    print(f'per-cell argmax      {base:.4f}', flush=True)
    print(f'band-structured      {sc(decode(og)):.4f}', flush=True)

    bias = [0., 0., 0., 0.]
    best = sc(decode(og, bias))
    for _ in range(3):
        for v in range(4):
            for d in (-0.5, -0.3, -0.18, -0.08, 0.08, 0.18, 0.3, 0.5):
                cand = list(bias); cand[v] += d
                s = sc(decode(og, cand))
                if s > best:
                    best, bias = s, cand
    print(f'+ tuned class bias   {best:.4f}  bias={[round(b,2) for b in bias]}', flush=True)
    json.dump({'bias': bias, 'oof': best}, open(W + r'\cache\struct_cfg.json', 'w'))
    print('saved struct_cfg.json', flush=True)


if __name__ == '__main__':
    main()
