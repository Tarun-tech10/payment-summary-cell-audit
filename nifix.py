"""num_items as an explicit counting problem.

The layout model already localises the item band; within it, each item contributes a line
with a right-aligned price.  Counting those lines is far more direct than asking a global
CNN pooling vector to regress a small integer, and the two disagree in different places, so
they blend well.
"""
import os, json, pickle, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from final2 import dg, NUM, C_LI, C_NLINES, C_YC, C_XC, C_RELX, C_RELW, C_LH

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
NI_MAX = 15


def band(gp):
    """Vertical extent of the item band and of the summary band, in [0,1]."""
    g = gp.reshape(4, 16).argmax(0).reshape(4, 4) if gp.ndim == 3 else gp.reshape(4, 4)
    r1 = np.where((g == 1).any(1))[0]
    r23 = np.where(np.isin(g, (2, 3)).any(1))[0]
    ract = np.where((g > 0).any(1))[0]
    if len(r1):
        y0, y1 = r1.min() / 4.0, (r1.max() + 1) / 4.0
    elif len(ract):
        y0, y1 = ract.min() / 4.0, (ract.max() + 1) / 4.0
    else:
        y0, y1 = 0.0, 1.0
    s0 = (r23.min() / 4.0) if len(r23) else 1.0
    return y0, min(y1, max(s0, y0 + 0.05)), s0


def feats(split, n, gp):
    lines = pickle.load(open(W + f'\\cache\\lines_{split}.pkl', 'rb'))
    pack = pickle.load(open(W + f'\\cache\\dec6_{split}.pkl', 'rb'))
    out = []
    for i in range(n):
        bx = np.asarray(lines[i][0], np.float32).reshape(-1, 4)
        y0, y1, s0 = band(gp[i])
        yc = (bx[:, 1] + bx[:, 3]) / 2 if len(bx) else np.zeros(0, np.float32)
        h = (bx[:, 3] - bx[:, 1]) if len(bx) else np.zeros(0, np.float32)
        wdt = (bx[:, 2] - bx[:, 0]) if len(bx) else np.zeros(0, np.float32)
        inb = (yc >= y0) & (yc < y1)
        mh = float(np.median(h)) if len(h) else 0.02
        # text content per detected line, from the scale-1.0 readings
        texts, confs, F = pack[i]
        nst = len(texts) // 3 if len(texts) and len(texts) % 3 == 0 else len(texts)
        byline = defaultdict(list)
        if nst:
            nl = max(F[0, C_NLINES] - 1, 1)
            li = np.round(F[:nst, C_LI] * nl).astype(int)
            for j in range(nst):
                byline[int(li[j])].append((texts[j], float(F[j, C_YC]),
                                           float(F[j, C_RELX] + F[j, C_RELW])))
        nnum = nprice = nright = 0
        for ln, items in byline.items():
            yy = float(np.mean([b for _, b, _ in items]))
            if not (y0 <= yy < y1):
                continue
            joined = ' '.join(t for t, _, _ in items)
            nums = [m for m in NUM.findall(joined) if len(dg(m)) >= 2]
            nnum += 1 if nums else 0
            nprice += 1 if any(len(dg(m)) >= 3 for m in nums) else 0
            nright += 1 if any(x > 0.72 for _, _, x in items) else 0
        span = max(y1 - y0, 1e-3)
        out.append([
            len(bx), float(inb.sum()), span, mh, span / max(mh, 1e-3),
            float(h[inb].mean()) if inb.any() else 0.0,
            float(wdt[inb].mean()) if inb.any() else 0.0,
            float(np.median(np.diff(np.sort(yc[inb])))) if inb.sum() > 2 else 0.0,
            span / max(float(np.median(np.diff(np.sort(yc[inb])))) if inb.sum() > 2 else mh, 1e-3),
            nnum, nprice, nright, len(byline), y0, y1, s0,
            float((yc < y0).sum()), float((yc >= y1).sum()),
            float((wdt[inb] > np.median(wdt) if len(wdt) else 0).sum()) if inb.any() else 0.0,
        ])
    return np.array(out, np.float32)


def cnn_tags(kind='n'):
    """Only tags with BOTH an oof and a test file -- otherwise the matrices disagree."""
    return [t for t in ['', '2', '3', '4']
            if os.path.exists(W + f'\\cache\\oof{t}_{kind}.npy')
            and os.path.exists(W + f'\\cache\\te{t}_{kind}.npy')]


def cnnprobs(split, tags):
    pre = 'oof' if split == 'train' else 'te'
    return [np.load(W + f'\\cache\\{pre}{t}_n.npy') for t in tags]


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    y = np.array([min(a['num_items'], NI_MAX) for a in A], np.int64)
    n, nt = len(tr), len(te)
    og = np.load(W + r'\cache\oofC_g.npy'); tg = np.load(W + r'\cache\teC_g.npy')
    Ftr = feats('train', n, og); Fte = feats('test', nt, tg)
    tags = cnn_tags('n')
    ptr = cnnprobs('train', tags); pte = cnnprobs('test', tags)
    print(f'cnn num_items sources {tags}', flush=True)
    for k, p in enumerate(ptr):
        print(f'  source {k} acc {(p.argmax(1) == y).mean():.4f}')
    avg = np.mean(ptr, 0)
    print(f'  cnn average acc {(avg.argmax(1) == y).mean():.4f}', flush=True)

    Xtr = np.concatenate([Ftr] + ptr, 1)
    Xte = np.concatenate([Fte] + pte, 1)
    P = dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=0.04,
             num_leaves=31, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
             min_data_in_leaf=20, verbose=-1, seed=0)
    oof = np.zeros((n, NI_MAX + 1), np.float32)
    pt = np.zeros((nt, NI_MAX + 1), np.float32)
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, np.minimum(y, 8)):
        m = lgb.train(P, lgb.Dataset(Xtr[ti], y[ti]), num_boost_round=350)
        oof[vi] = m.predict(Xtr[vi]); pt += m.predict(Xte) / 5
    print(f'lgb(lines+cnn) acc {(oof.argmax(1) == y).mean():.4f}', flush=True)
    best, bw = None, None
    for w in np.arange(0, 1.01, 0.1):
        a = ((w * oof + (1 - w) * avg).argmax(1) == y).mean()
        print(f'  blend w={w:.1f}: {a:.4f}')
        if best is None or a > best:
            best, bw = a, w
    print(f'BEST num_items {best:.4f} at w={bw:.1f}', flush=True)
    np.save(W + r'\cache\ni_lgb_oof.npy', oof); np.save(W + r'\cache\ni_lgb_te.npy', pt)


if __name__ == '__main__':
    main()
