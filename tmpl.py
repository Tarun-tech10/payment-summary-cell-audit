"""Is the 4x4 gold grid drawn from a small set of structural templates?

The gold is a rasterisation of stacked text-block rectangles, so most of the 4^16 label
space can never occur.  If the observed set is small, decoding becomes 'pick a template'
and we can do a proper MBR decode against the real metric instead of per-cell argmax.
"""
import json
from collections import Counter
import numpy as np, pandas as pd
from scorer import grid_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'

tr = pd.read_csv(D + r'\train.csv')
A = [json.loads(x) for x in tr.answer_json]
G = np.array([a['semantic_grid'] for a in A], np.int64)
n = len(G)
cnt = Counter(map(tuple, G))
print(f'{n} images, {len(cnt)} unique grids')
cov = np.cumsum([c for _, c in cnt.most_common()]) / n
for k in (1, 5, 10, 20, 50, 100, 150):
    if k <= len(cov):
        print(f'  top {k:3d} templates cover {cov[k-1]:.3f}')

# structural check: is each class a contiguous band of rows with a shared column span?
rowuni = colspan = ordered = 0
for g in G:
    m = g.reshape(4, 4)
    rowuni += all(len(set(r[r > 0].tolist())) <= 1 for r in m)
    act = m > 0
    cols = [tuple(np.where(r)[0].tolist()) for r in act if r.any()]
    colspan += len(set(cols)) <= 1
    seq = [v for r in m for v in r if v > 0]
    ordered += all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
print(f'rows class-uniform {rowuni/n:.3f} | identical col span {colspan/n:.3f} | '
      f'classes non-decreasing top-to-bottom {ordered/n:.3f}')

H = np.array([list(k) for k, _ in cnt.most_common()], np.int64)
prior = np.array([c for _, c in cnt.most_common()], np.float64)
prior /= prior.sum()
print(f'hypothesis set {H.shape}')

# pairwise metric matrix between templates: S[a,b] = grid_score(H[a], H[b])
S = np.array([[grid_score(H[a], H[b]) for b in range(len(H))] for a in range(len(H))])
print('metric matrix', S.shape)

# oracle: best template available for each gold
idx = {tuple(h): i for i, h in enumerate(H)}
gi = np.array([idx[tuple(g)] for g in G])
print(f'oracle best-template score {S[:, gi].max(0).mean():.4f} (gold is in the set by construction)')

np.save(W + r'\cache\tmpl_H.npy', H)
np.save(W + r'\cache\tmpl_prior.npy', prior)
np.save(W + r'\cache\tmpl_S.npy', S)


def mbr(P4, S, H, prior, temp=1.0, alpha=1.0):
    """P4: (n,4,4,4) class-prob maps. Returns chosen 16-vectors."""
    logp = np.log(np.clip(P4.reshape(len(P4), 4, 16), 1e-6, None))
    # loglik of each hypothesis under the per-cell marginals
    ll = np.stack([logp[:, H[h], np.arange(16)].sum(1) for h in range(len(H))], 1)
    ll = ll / temp + alpha * np.log(prior)[None, :]
    wgt = np.exp(ll - ll.max(1, keepdims=True))
    wgt /= wgt.sum(1, keepdims=True)
    return (wgt @ S.T).argmax(1)


if __name__ == '__main__':
    og = np.load(W + r'\cache\oofC_g.npy')
    if og.ndim == 4 and og.shape[1] == 4 and og.shape[2] == 4:
        P4 = og if og.shape[3] == 4 else og.transpose(0, 2, 3, 1)
    print('oofC_g shape', og.shape)
    P = og.reshape(n, 4, 16) if og.shape[1] == 4 else None
    base = np.array([grid_score(P[i].argmax(0), G[i]) for i in range(n)]).mean()
    print(f'argmax baseline {base:.4f}')
    for temp in (0.5, 1.0, 2.0, 3.0, 5.0):
        for alpha in (0.0, 0.5, 1.0):
            ch = mbr(og.reshape(n, 4, 4, 4), S, H, prior, temp, alpha)
            sc = np.array([grid_score(H[ch[i]], G[i]) for i in range(n)]).mean()
            print(f'  MBR temp={temp} alpha={alpha}: {sc:.4f}')
