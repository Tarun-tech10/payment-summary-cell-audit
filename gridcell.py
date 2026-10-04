"""Per-cell grid refinement: CNN probabilities + OCR keyword evidence in that cell.

The layout network cannot read, but 'subtotal area' vs 'final-total area' is a semantic
distinction (SUBTOTAL/PPN/SERVICE vs TOTAL).  Feeding per-cell keyword evidence to a
gradient-boosted per-cell classifier targets exactly the class-2 confusion that dominates
the macro-IoU loss.
"""
import os, json, pickle, re, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scorer import grid_score
from wordfeat import token_keyscores, KEYS

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
KL = list(KEYS)
C_KIND, C_YC, C_XC, C_LH, C_NLINES, C_RIGHT = 6, 16, 17, 18, 19, 11
NUM = re.compile(r'-?\d[\d.,]*\d|\d')


def cell_ocr_feats(row):
    """(16, nf) OCR evidence per grid cell."""
    texts, confs, F = row
    nf = 6 + 2 * len(KL)
    out = np.zeros((16, nf), np.float32)
    if len(texts) == 0:
        return out
    S, _ = token_keyscores(texts)
    r = np.clip((F[:, C_YC] * 4).astype(int), 0, 3)
    c = np.clip((F[:, C_XC] * 4).astype(int), 0, 3)
    cell = r * 4 + c
    ndig = np.array([len(re.sub(r'\D', '', t)) for t in texts], np.float32)
    for k in range(16):
        m = cell == k
        if not m.any():
            continue
        v = [float(m.sum()), float((ndig[m] >= 3).sum()), float(np.max(confs[m])),
             float(np.mean(confs[m])), float(np.max(F[m, C_LH])),
             float((F[m, C_RIGHT] > 0).sum())]
        out[k] = np.array(v + list(S[m].max(0)) + list((S[m] > .85).sum(0)), np.float32)
    return out


def build(split, n, gprob, fprob):
    dec = pickle.load(open(W + f'\\cache\\{split}', 'rb')) if os.path.sep in split else None
    raise RuntimeError


def make_matrix(dec, gprob, fprob, n):
    rows = []
    for i in range(n):
        oc = cell_ocr_feats(dec[i])
        P = gprob[i].reshape(4, 16)             # class x cell
        for k in range(16):
            r, c = divmod(k, 4)
            up = P[:, k - 4] if r > 0 else np.zeros(4, np.float32)
            dn = P[:, k + 4] if r < 3 else np.zeros(4, np.float32)
            lf = P[:, k - 1] if c > 0 else np.zeros(4, np.float32)
            rt = P[:, k + 1] if c < 3 else np.zeros(4, np.float32)
            rows.append(np.concatenate([
                P[:, k], up, dn, lf, rt,
                P.max(1), P[:, k] - P.max(1),
                [r, c, float(r) / 3, float(c) / 3],
                oc[k], oc.sum(0), fprob[i]]))
    return np.array(rows, np.float32)


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    G = np.array([a['semantic_grid'] for a in A])
    ntr, nte = len(tr), len(te)
    og = (2 * np.load(W + r'\cache\oof_g.npy') + np.load(W + r'\cache\oof2_g.npy')
          + np.load(W + r'\cache\oof3_g.npy')) / 4
    tg = (2 * np.load(W + r'\cache\te1_g.npy') + np.load(W + r'\cache\te2_g.npy')
          + np.load(W + r'\cache\te3_g.npy')) / 4
    of = np.load(W + r'\cache\oof2_f.npy'); tf = np.load(W + r'\cache\te2_f.npy')

    def pick_dec(split):
        for nm in ['dec8', 'dec6', 'dec7', 'dec4', 'dec5', 'dec3', 'dec2']:
            p = W + f'\\cache\\{nm}_{split}.pkl'
            if os.path.exists(p):
                return pickle.load(open(p, 'rb'))
        raise FileNotFoundError(split)
    dtr, dte = pick_dec('train'), pick_dec('test')

    Xtr = make_matrix(dtr, og, of, ntr)
    Xte = make_matrix(dte, tg, tf, nte)
    y = G.ravel()
    grp = np.repeat(np.arange(ntr), 16)
    print('cell matrix', Xtr.shape, flush=True)
    P = dict(objective='multiclass', num_class=4, learning_rate=0.05, num_leaves=63,
             min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, verbose=-1, seed=0)
    oof = np.zeros((len(y), 4), np.float32)
    for ti, vi in GroupKFold(5).split(Xtr, y, grp):
        m = lgb.train(P, lgb.Dataset(Xtr[ti], y[ti]), num_boost_round=400)
        oof[vi] = m.predict(Xtr[vi])
    oof4 = oof.reshape(ntr, 16, 4).transpose(0, 2, 1).reshape(ntr, 4, 4, 4)
    base = float(np.mean([grid_score(og[i].argmax(0).ravel(), G[i]) for i in range(ntr)]))
    lgbs = float(np.mean([grid_score(oof4[i].argmax(0).ravel(), G[i]) for i in range(ntr)]))
    print(f'grid: cnn {base:.4f}  cell-lgbm {lgbs:.4f}', flush=True)
    best = (base, 0.0)
    for w in np.arange(0, 1.01, 0.1):
        Q = w * oof4 + (1 - w) * og
        s = float(np.mean([grid_score(Q[i].argmax(0).ravel(), G[i]) for i in range(ntr)]))
        if s > best[0]:
            best = (s, float(w))
    print(f'best blend w={best[1]:.2f} -> {best[0]:.4f}', flush=True)
    mfull = lgb.train(P, lgb.Dataset(Xtr, y), num_boost_round=400)
    pte = mfull.predict(Xte).reshape(nte, 16, 4).transpose(0, 2, 1).reshape(nte, 4, 4, 4)
    np.save(W + r'\cache\oofC_g.npy', best[1] * oof4 + (1 - best[1]) * og)
    np.save(W + r'\cache\teC_g.npy', best[1] * pte + (1 - best[1]) * tg)
    print('saved blended grid probs', flush=True)


if __name__ == '__main__':
    main()
