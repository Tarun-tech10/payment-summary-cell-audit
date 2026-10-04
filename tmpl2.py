"""Honest MBR-over-templates: the hypothesis set must be built out-of-fold.

Also tries enlarging the set with structural variants so a held-out layout that was never
observed verbatim still has a close neighbour to decode to.
"""
import json, sys
from collections import Counter
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def shift_grid(g, dr, dc):
    m = g.reshape(4, 4); o = np.zeros((4, 4), np.int64)
    r0, r1 = max(0, dr), min(4, 4 + dr)
    c0, c1 = max(0, dc), min(4, 4 + dc)
    o[r0:r1, c0:c1] = m[max(0, -dr):min(4, 4 - dr), max(0, -dc):min(4, 4 - dc)]
    return o.ravel()


def enlarge(base):
    """Observed layouts plus one-cell translations -- a shifted receipt is still a receipt."""
    out = {tuple(g) for g in base}
    for g in base:
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            s = shift_grid(g, dr, dc)
            if (s > 0).sum() >= 2 and (s == 3).any():
                out.add(tuple(s))
    return np.array(sorted(out), np.int64)


def mbr_choose(P4, H, S, prior, temp=3.0, alpha=0.0):
    logp = np.log(np.clip(P4.reshape(len(P4), 4, 16), 1e-6, None))
    ar = np.arange(16)
    ll = np.stack([logp[:, H[h], ar].sum(1) for h in range(len(H))], 1)
    ll = ll / temp + alpha * np.log(prior)[None, :]
    w = np.exp(ll - ll.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return (w @ S.T).argmax(1)


def pair_matrix(A, B):
    return np.array([[grid_score(a, b) for b in B] for a in A])


def main():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    G = np.array([a['semantic_grid'] for a in A], np.int64)
    ni = np.array([min(a['num_items'], 8) for a in A])
    n = len(G)
    og = np.load(W + r'\cache\oofC_g.npy')
    base = np.mean([grid_score(og[i].reshape(4, 16).argmax(0), G[i]) for i in range(n)])
    print(f'argmax baseline {base:.4f}', flush=True)

    for mode in ('observed', 'enlarged'):
        for temp in (1.0, 3.0, 6.0):
            pred = np.zeros((n, 16), np.int64); cover = 0
            for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), ni):
                cnt = Counter(map(tuple, G[ti]))
                Hs = np.array([list(k) for k, _ in cnt.most_common()], np.int64)
                pr = np.array([c for _, c in cnt.most_common()], np.float64); pr /= pr.sum()
                if mode == 'enlarged':
                    Hs2 = enlarge(Hs)
                    seen = {tuple(h): p for h, p in zip(Hs, pr)}
                    pr = np.array([seen.get(tuple(h), pr.min() * 0.5) for h in Hs2])
                    pr /= pr.sum(); Hs = Hs2
                S = pair_matrix(Hs, Hs)
                ch = mbr_choose(og[vi], Hs, S, pr, temp=temp)
                pred[vi] = Hs[ch]
                cover += sum(tuple(g) in {tuple(h) for h in Hs} for g in G[vi])
            sc = np.mean([grid_score(pred[i], G[i]) for i in range(n)])
            print(f'{mode:9s} temp={temp}: {sc:.4f}   (held-out gold present in H: {cover/n:.3f})',
                  flush=True)


if __name__ == '__main__':
    main()
