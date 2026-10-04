"""Two-stage answer selection: pick the digits first, then the printed format.

Reading digits and reproducing punctuation are different problems -- digits carry the
evidence, formatting is nearly deterministic given the receipt's own style.  Ranking them
jointly makes the digit decision compete with format variants of the same number.
"""
import json, pickle, re, os
from collections import Counter, defaultdict
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scorer import total_price_score
from final2 import build, dg, group3

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def pattern_of(s):
    return ''.join('D' if c.isdigit() else c for c in s)


def digit_matrix(X, grp, cands, n):
    """Aggregate candidate rows to one row per (image, digit-string)."""
    keys, rows, owner = [], [], []
    idx_by = defaultdict(list)
    for r, (g, c) in enumerate(zip(grp, cands)):
        idx_by[(g, dg(c))].append(r)
    for (g, d), rs in idx_by.items():
        sub = X[rs]
        rows.append(np.concatenate([sub.max(0), sub.mean(0), [len(rs), len(d), float(d)
                                    if len(d) <= 12 else 0.0, np.log1p(float(d) if len(d) <= 12 else 0)]]))
        keys.append((g, d)); owner.append(g)
    return np.array(rows, np.float32), np.array(owner), keys


def main():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    n = len(A)
    X, grp, cands, lab = build('train', n, golds, use_beam=False)
    print('cands', X.shape, flush=True)
    hit = np.zeros(n, bool)
    for g, l in zip(grp, lab):
        if l > 0:
            hit[g] = True
    print('string recall ceiling', round(hit.mean(), 4), flush=True)

    # ---------- stage A: digits
    XD, ownD, keys = digit_matrix(X, grp, cands, n)
    labD = np.array([1.0 if d == dg(golds[g]) else 0.0 for (g, d) in keys], np.float32)
    dhit = np.zeros(n, bool)
    for (g, d), l in zip(keys, labD):
        if l > 0:
            dhit[g] = True
    print('digit recall ceiling', round(dhit.mean(), 4), 'rows', XD.shape, flush=True)
    P = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
    oofD = np.zeros(len(XD), np.float32)
    for ti, vi in GroupKFold(5).split(XD, labD, ownD):
        m = lgb.train(P, lgb.Dataset(XD[ti], labD[ti]), num_boost_round=500)
        oofD[vi] = m.predict(XD[vi])
    bestd = {}
    for r, (g, d) in enumerate(keys):
        if g not in bestd or oofD[r] > bestd[g][1]:
            bestd[g] = (d, oofD[r])
    dacc = float(np.mean([bestd.get(i, ('', 0))[0] == dg(golds[i]) for i in range(n)]))
    print(f'stage-A digit accuracy {dacc:.4f}  (efficiency {dacc/max(dhit.mean(),1e-9):.3f})', flush=True)

    # ---------- stage B: format, restricted to the chosen digits
    pri = Counter(pattern_of(g) for g in golds)
    tot = sum(pri.values())
    keep = [r for r, c in enumerate(cands) if dg(c) == bestd.get(grp[r], ('', 0))[0]]
    XB = np.concatenate([X[keep], np.array([[
        np.log((pri.get(pattern_of(cands[r]), 0) + 0.5) / (tot + 50.0)),
        len(cands[r]), 1.0 if cands[r] == dg(cands[r]) else 0.0] for r in keep], np.float32)], 1)
    gB = grp[keep]; cB = [cands[r] for r in keep]
    labB = np.array([1.0 if cands[r] == golds[grp[r]] else 0.0 for r in keep], np.float32)
    oofB = np.zeros(len(XB), np.float32)
    for ti, vi in GroupKFold(5).split(XB, labB, gB):
        m = lgb.train(P, lgb.Dataset(XB[ti], labB[ti]), num_boost_round=400)
        oofB[vi] = m.predict(XB[vi])
    pick2 = {}
    for r in range(len(XB)):
        g = gB[r]
        if g not in pick2 or oofB[r] > pick2[g][1]:
            pick2[g] = (cB[r], oofB[r])
    pred = [pick2.get(i, ('25.000', 0))[0] for i in range(n)]
    ex = float(np.mean([pred[i] == golds[i] for i in range(n)]))
    sc = float(np.mean([total_price_score(pred[i], golds[i]) for i in range(n)]))
    print(f'TWO-STAGE  exact {ex:.4f}  total_price {sc:.4f}', flush=True)

    # ---------- flat baseline for comparison
    oof = np.zeros(len(X), np.float32)
    for ti, vi in GroupKFold(5).split(X, lab, grp):
        m = lgb.train(P, lgb.Dataset(X[ti], lab[ti]), num_boost_round=600)
        oof[vi] = m.predict(X[vi])
    flat = {}
    for r in range(len(X)):
        g = grp[r]
        if g not in flat or oof[r] > flat[g][1]:
            flat[g] = (cands[r], oof[r])
    fp = [flat.get(i, ('25.000', 0))[0] for i in range(n)]
    fex = float(np.mean([fp[i] == golds[i] for i in range(n)]))
    fsc = float(np.mean([total_price_score(fp[i], golds[i]) for i in range(n)]))
    print(f'FLAT       exact {fex:.4f}  total_price {fsc:.4f}', flush=True)


if __name__ == '__main__':
    main()
