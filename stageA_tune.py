"""Sweep stage-A (digit-selection) hyperparameters -- efficiency there caps exact match."""
import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from final2 import build, dg, digit_matrix

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
tr = pd.read_csv(D + r'\train.csv')
A = [json.loads(x) for x in tr.answer_json]
golds = [a['total_price'].strip() for a in A]
n = len(A)
X, grp, cands, lab = build('train', n, golds, use_beam=False)
XD, ownD, keys = digit_matrix(X, grp, cands)
labD = np.array([1.0 if d == dg(golds[g]) else 0.0 for (g, d) in keys], np.float32)
dhit = np.zeros(n, bool)
for (g, d), l in zip(keys, labD):
    if l > 0:
        dhit[g] = True
print('digit ceiling', round(dhit.mean(), 4), 'rows', XD.shape, flush=True)


def run(P, rounds):
    oof = np.zeros(len(XD), np.float32)
    for ti, vi in GroupKFold(5).split(XD, labD, ownD):
        oof[vi] = lgb.train(P, lgb.Dataset(XD[ti], labD[ti]), num_boost_round=rounds).predict(XD[vi])
    best = {}
    for r, (g, d) in enumerate(keys):
        if g not in best or oof[r] > best[g][1]:
            best[g] = (d, oof[r])
    acc = float(np.mean([best.get(i, ('', 0))[0] == dg(golds[i]) for i in range(n)]))
    return acc, acc / max(dhit.mean(), 1e-9)


base = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.7,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
for name, P, R in [
    ('base/500', base, 500),
    ('deep/900', dict(base, num_leaves=127, min_data_in_leaf=15, learning_rate=0.035), 900),
    ('shallow/900', dict(base, num_leaves=31, min_data_in_leaf=60, learning_rate=0.04), 900),
    ('ff0.5/700', dict(base, feature_fraction=0.5), 700),
    ('lambda/700', dict(base, lambda_l2=5.0, num_leaves=95), 700),
]:
    a, e = run(P, R)
    print(f'{name:14s} digit acc {a:.4f}  efficiency {e:.3f}', flush=True)
