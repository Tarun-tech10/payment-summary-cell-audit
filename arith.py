"""Does receipt arithmetic identify the total?

A receipt is an internally consistent little ledger: the total is the number that equals
subtotal + tax + service, and the one that cash minus change lands on.  Those relations hold
between candidate VALUES, so they are checkable without reading any label -- and they say
something no per-strip confidence feature can.
"""
import json, sys
from collections import defaultdict
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from final2 import build, dg, digit_matrix

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def stageA(XD, labD, ownD, keys, golds, n, rounds=900):
    P = dict(objective='binary', learning_rate=0.035, num_leaves=127, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=15, verbose=-1, seed=0)
    oof = np.zeros(len(XD), np.float32)
    for ti, vi in GroupKFold(5).split(XD, labD, ownD):
        oof[vi] = lgb.train(P, lgb.Dataset(XD[ti], labD[ti]), num_boost_round=rounds).predict(XD[vi])
    best = {}
    for r, (g, d) in enumerate(keys):
        if g not in best or oof[r] > best[g][1]:
            best[g] = (d, oof[r])
    return float(np.mean([best.get(i, ('', 0))[0] == dg(golds[i]) for i in range(n)]))


def main():
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    n = len(golds)
    X, grp, cands, lab = build('train', n, golds, use_beam=False)
    XA, ownD, keys = digit_matrix(X, grp, cands, use_arith=True)
    X0 = digit_matrix(X, grp, cands, use_arith=False)[0]
    labD = np.array([1.0 if d == dg(golds[g]) else 0.0 for (g, d) in keys], np.float32)
    ceil = np.zeros(n, bool)
    for (g, d), l in zip(keys, labD):
        if l > 0:
            ceil[g] = True
    print(f'digit rows {XA.shape}  ceiling {ceil.mean():.4f}', flush=True)
    a0 = stageA(X0, labD, ownD, keys, golds, n)
    print(f'stage A without arithmetic: {a0:.4f}  (efficiency {a0/ceil.mean():.3f})', flush=True)
    a1 = stageA(XA, labD, ownD, keys, golds, n)
    print(f'stage A with arithmetic:    {a1:.4f}  (efficiency {a1/ceil.mean():.3f})', flush=True)


if __name__ == '__main__':
    main()
