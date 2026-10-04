"""Given the CORRECT digits, how often do we recover the printed form? Isolates stage B."""
import json, re
from collections import Counter
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from final2 import build, dg, group3

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
tr = pd.read_csv(D + r'\train.csv')
A = [json.loads(x) for x in tr.answer_json]
golds = [a['total_price'].strip() for a in A]
n = len(A)
X, grp, cands, lab = build('train', n, golds, use_beam=False)


def pat(s):
    return ''.join('D' if c.isdigit() else c for c in s)


# oracle digits: restrict to candidates whose digits match the gold digits
keep = [r for r in range(len(cands)) if dg(cands[r]) == dg(golds[grp[r]])]
gB = grp[keep]; cB = [cands[r] for r in keep]
have = set(gB.tolist())
print('images where correct digits are available:', round(len(have) / n, 4))
reach = sum(1 for i in range(n) if any(cB[j] == golds[i] for j in range(len(cB)) if gB[j] == i))
print('of those, printed form present:', round(reach / max(len(have), 1), 4), flush=True)

# how good is "most common pattern among same-length numbers on this receipt"?
pri = Counter(pat(g) for g in golds); tot = sum(pri.values())
recpat = {}
for i in range(n):
    m = grp == i
    c = Counter(pat(cands[r]) for r in np.where(m)[0]
                if len(dg(cands[r])) == len(dg(golds[i])))
    recpat[i] = c

hit_prior = hit_rec = 0
for i in have:
    opts = [cB[j] for j in range(len(cB)) if gB[j] == i]
    if not opts:
        continue
    best = max(opts, key=lambda s: pri.get(pat(s), 0))
    hit_prior += best == golds[i]
    best2 = max(opts, key=lambda s: (recpat[i].get(pat(s), 0), pri.get(pat(s), 0)))
    hit_rec += best2 == golds[i]
print('global-pattern-prior rule:', round(hit_prior / len(have), 4))
print('receipt-modal-pattern rule:', round(hit_rec / len(have), 4), flush=True)

# learned stage B on oracle digits
ex = np.array([[np.log((pri.get(pat(cB[j]), 0) + .5) / (tot + 50.)),
                len(cB[j]), 1.0 if cB[j] == dg(cB[j]) else 0.0,
                float(recpat[gB[j]].get(pat(cB[j]), 0)),
                float(recpat[gB[j]].get(pat(cB[j]), 0)) / max(sum(recpat[gB[j]].values()), 1),
                1.0 if pat(cB[j]) == (recpat[gB[j]].most_common(1)[0][0]
                                      if recpat[gB[j]] else '') else 0.0]
               for j in range(len(cB))], np.float32)
XB = np.concatenate([X[keep], ex], 1)
labB = np.array([1.0 if cB[j] == golds[gB[j]] else 0.0 for j in range(len(cB))], np.float32)
P = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
oof = np.zeros(len(XB), np.float32)
for ti, vi in GroupKFold(5).split(XB, labB, gB):
    oof[vi] = lgb.train(P, lgb.Dataset(XB[ti], labB[ti]), num_boost_round=400).predict(XB[vi])
best = {}
for j in range(len(cB)):
    g = gB[j]
    if g not in best or oof[j] > best[g][1]:
        best[g] = (cB[j], oof[j])
acc = np.mean([best[i][0] == golds[i] for i in have])
print('learned stage B on oracle digits:', round(float(acc), 4), flush=True)
