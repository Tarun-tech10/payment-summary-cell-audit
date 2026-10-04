"""Class-presence gating for the grid.

macro-IoU averages over every class present in EITHER the prediction or the gold, so a class
predicted that the gold does not have contributes a whole zero term -- on a typical
three-class receipt that is a quarter of the grid score.  Presence is a global property, so
it gets its own image-level classifier and its own decision threshold, tuned directly
against the metric rather than left to per-cell argmax.
"""
import os, json, pickle, re, sys
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score
from wordfeat import token_keyscores, KEYS

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
KL = list(KEYS)


def img_feats(dec_row, gp, fp):
    texts, confs, F = dec_row
    P = gp.reshape(4, 16)
    am = P.argmax(0)
    f = []
    for v in range(4):
        s = np.sort(P[v])[::-1]
        f += [float(P[v].sum()), float(P[v].max()), float((am == v).sum()),
              float(s[:3].mean()), float(s[:6].mean()), float(np.median(P[v]))]
    f += list(fp) + [float(fp.sum()), float(fp.max())]
    if len(texts):
        S, _ = token_keyscores(texts)
        f += list(S.max(0)) + list((S > .85).sum(0)) + [float(len(texts)), float(F[0, 19])]
        nd = np.array([len(re.sub(r'\D', '', t)) for t in texts], np.float32)
        f += [float((nd >= 3).sum()), float((nd >= 5).sum()), float(np.mean(confs))]
    else:
        f += [0.0] * (2 * len(KL) + 5)
    return np.array(f, np.float32)


def biased(P4, b):
    """Per-class log-prob offsets, tuned against the metric rather than left at 0."""
    return np.log(np.clip(P4.reshape(len(P4), 4, 16), 1e-9, None)) + np.asarray(b, np.float32)[None, :, None]


def decode(Q, pres, thr, hi):
    """argmax over biased log-probs, but a class is only allowed on the map if the
    image-level gate says it is present."""
    out = []
    NEG = -1e9
    for i in range(len(Q)):
        q = Q[i].copy()
        for v in (1, 2, 3):
            if pres[i, v] < thr[v]:
                q[v] = NEG
        p = q.argmax(0)
        for v in (1, 2, 3):
            if pres[i, v] >= hi[v] and not (p == v).any():
                p[int(np.argmax(Q[i][v]))] = v
        if not (p == 3).any():
            p[int(np.argmax(Q[i][3]))] = 3
        out.append(p)
    return np.array(out)


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    G = np.array([a['semantic_grid'] for a in A])
    ntr, nte = len(tr), len(te)
    og = np.load(W + r'\cache\oofC_g.npy'); tg = np.load(W + r'\cache\teC_g.npy')
    of = np.load(W + r'\cache\oof2_f.npy'); tf = np.load(W + r'\cache\te2_f.npy')
    dtr = pickle.load(open(W + r'\cache\dec6_train.pkl', 'rb'))
    dte = pickle.load(open(W + r'\cache\dec6_test.pkl', 'rb'))

    Y = np.array([[(G[i] == v).any() for v in range(4)] for i in range(ntr)], np.float32)
    print('gold class presence rates', Y.mean(0).round(3), flush=True)
    amp = og.reshape(ntr, 4, 16).argmax(1)
    for v in (1, 2, 3):
        print(f'  class {v}: argmax presence agrees {np.mean((amp == v).any(1) == (Y[:, v] > .5)):.4f}')

    Xtr = np.stack([img_feats(dtr[i], og[i], of[i]) for i in range(ntr)])
    Xte = np.stack([img_feats(dte[i], tg[i], tf[i]) for i in range(nte)])
    print('presence matrix', Xtr.shape, flush=True)
    P = dict(objective='binary', learning_rate=0.04, num_leaves=15, min_data_in_leaf=20,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    pres = np.zeros((ntr, 4), np.float32); prte = np.zeros((nte, 4), np.float32)
    pres[:, 0] = prte[:, 0] = 1.0
    for v in (1, 2, 3):
        if Y[:, v].min() == Y[:, v].max():
            pres[:, v] = prte[:, v] = Y[0, v]
            continue
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, Y[:, v]):
            pres[vi, v] = lgb.train(P, lgb.Dataset(Xtr[ti], Y[ti, v]),
                                    num_boost_round=300).predict(Xtr[vi])
        prte[:, v] = lgb.train(P, lgb.Dataset(Xtr, Y[:, v]), num_boost_round=300).predict(Xte)
        auc_acc = np.mean((pres[:, v] > .5) == (Y[:, v] > .5))
        print(f'  class {v}: learned presence acc {auc_acc:.4f}', flush=True)

    base = float(np.mean([grid_score(og[i].reshape(4, 16).argmax(0), G[i]) for i in range(ntr)]))
    print(f'argmax baseline {base:.4f}', flush=True)

    def sc(p):
        return float(np.mean([grid_score(p[i], G[i]) for i in range(ntr)]))

    # ---- per-class log-prob bias, coordinate ascent against the metric
    off = {1: 2.0, 2: 2.0, 3: 2.0}
    bias = [0.0, 0.0, 0.0, 0.0]
    bbest = sc(decode(biased(og, bias), pres, {1: 0., 2: 0., 3: 0.}, off))
    for _ in range(2):
        for v in range(4):
            for d in (-0.6, -0.4, -0.25, -0.12, 0.12, 0.25, 0.4, 0.6):
                cand = list(bias); cand[v] += d
                s = sc(decode(biased(og, cand), pres, {1: 0., 2: 0., 3: 0.}, off))
                if s > bbest:
                    bbest, bias = s, cand
    print(f'class bias {[round(b,2) for b in bias]} -> {bbest:.4f}', flush=True)
    Q = biased(og, bias)

    # ---- presence gate on top
    best = (bbest, {1: 0.0, 2: 0.0, 3: 0.0}, dict(off))
    for v in (1, 2, 3):
        for t in [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]:
            for h in [2.0, 0.85, 0.7, 0.55, 0.4]:
                t2 = dict(best[1]); h2 = dict(best[2]); t2[v] = t; h2[v] = h
                s = sc(decode(Q, pres, t2, h2))
                if s > best[0]:
                    best = (s, t2, h2)
        print(f'  after class {v}: {best[0]:.4f} thr={best[1]} hi={best[2]}', flush=True)
    print(f'GATED GRID {best[0]:.4f} (from {base:.4f})', flush=True)
    np.save(W + r'\cache\pres_oof.npy', pres); np.save(W + r'\cache\pres_te.npy', prte)
    json.dump({'thr': {str(k): v for k, v in best[1].items()},
               'hi': {str(k): v for k, v in best[2].items()},
               'bias': bias, 'oof': best[0]}, open(W + r'\cache\pres_thr.json', 'w'))
    print('saved presence gate', flush=True)


if __name__ == '__main__':
    main()
