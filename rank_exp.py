"""Selection-efficiency experiments on the existing candidate set (CPU only)."""
import json, pickle, re, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scorer import total_price_score
from final import build, pick

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'

tr = pd.read_csv(D + r'\train.csv')
A = [json.loads(x) for x in tr.answer_json]
golds = [a['total_price'].strip() for a in A]
n = len(tr)
X, grp, cands, lab = build('train', n, golds)
hit = np.zeros(n, bool)
for g, l in zip(grp, lab):
    if l > 0:
        hit[g] = True
print('recall ceiling', round(hit.mean(), 4), 'cands', X.shape)


def evaluate(pred):
    tot = pick(pred, grp, cands, n)
    ex = float(np.mean([tot[i] == golds[i] for i in range(n)]))
    sc = float(np.mean([total_price_score(tot[i], golds[i]) for i in range(n)]))
    return ex, sc, ex / max(hit.mean(), 1e-9)


def cv_binary(params, rounds):
    oof = np.zeros(len(X), np.float32)
    for ti, vi in GroupKFold(5).split(X, lab, grp):
        m = lgb.train(params, lgb.Dataset(X[ti], lab[ti]), num_boost_round=rounds)
        oof[vi] = m.predict(X[vi])
    return oof


def cv_rank(params, rounds):
    oof = np.zeros(len(X), np.float32)
    for ti, vi in GroupKFold(5).split(X, lab, grp):
        # lambdarank needs contiguous groups
        o = np.argsort(grp[ti], kind='stable'); tix = ti[o]
        gs = pd.Series(grp[tix]).value_counts(sort=False).reindex(
            pd.unique(grp[tix])).values
        d = lgb.Dataset(X[tix], lab[tix], group=gs)
        m = lgb.train(params, d, num_boost_round=rounds)
        oof[vi] = m.predict(X[vi])
    return oof


base = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.75,
            bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40, verbose=-1, seed=0)
print('binary/600      ', [round(v, 4) for v in evaluate(cv_binary(base, 600))])
b2 = dict(base, num_leaves=127, min_data_in_leaf=20, learning_rate=0.04)
print('binary/1200 wide', [round(v, 4) for v in evaluate(cv_binary(b2, 1200))])
rk = dict(objective='lambdarank', metric='ndcg', ndcg_eval_at=[1], learning_rate=0.05,
          num_leaves=63, feature_fraction=0.75, bagging_fraction=0.8, bagging_freq=1,
          min_data_in_leaf=40, verbose=-1, seed=0, label_gain=[0, 1])
print('lambdarank/600  ', [round(v, 4) for v in evaluate(cv_rank(rk, 600))])
# average of the two
p1 = cv_binary(base, 600); p2 = cv_rank(rk, 600)


def norm(p):
    q = p.copy()
    for i in range(n):
        m = grp == i
        if m.sum():
            v = q[m]
            q[m] = (v - v.mean()) / (v.std() + 1e-6)
    return q


print('blend           ', [round(v, 4) for v in evaluate(norm(p1) + norm(p2))])
