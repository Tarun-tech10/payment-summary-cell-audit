"""Pick which decoded token is the printed final total."""
import os, json, pickle, re, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scorer import total_price_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def value_of(s):
    d = re.sub(r'\D', '', s)
    return float(d) if d else -1.0


def str_feats(texts, confs, feats):
    """Extra per-candidate features derived from the decoded strings."""
    n = len(texts)
    if n == 0:
        return np.zeros((0, feats.shape[1] + 16), np.float32)
    vals = np.array([value_of(t) for t in texts], np.float32)
    ndig = np.array([len(re.sub(r'\D', '', t)) for t in texts], np.float32)
    hassep = np.array([1.0 if re.search(r'[.,]', t) else 0.0 for t in texts], np.float32)
    isneg = np.array([1.0 if t.startswith('-') else 0.0 for t in texts], np.float32)
    from collections import Counter
    cnt = Counter(texts)
    dup = np.array([cnt[t] for t in texts], np.float32)
    vcnt = Counter(vals.tolist())
    vdup = np.array([vcnt[v] for v in vals], np.float32)
    mx = max(vals.max(), 1.0)
    rank = np.argsort(np.argsort(-vals)).astype(np.float32)
    # trailing zeros
    tz = np.array([len(re.sub(r'\D', '', t)) - len(re.sub(r'\D', '', t).rstrip('0'))
                   for t in texts], np.float32)
    y = feats[:, 0]  # rel y in roi
    below = np.array([(y > y[i]).sum() for i in range(n)], np.float32)
    ex = np.stack([
        vals / mx, np.log1p(np.maximum(vals, 0)), ndig, hassep, isneg, dup, vdup,
        rank, rank / max(n - 1, 1), tz, np.float32(n) * np.ones(n, np.float32),
        confs, below / max(n, 1),
        (vals == vals.max()).astype(np.float32),
        np.array([1.0 if re.fullmatch(r'-?\d{1,3}([.,]\d{3})*([.,]\d{2})?', t) else 0.0
                  for t in texts], np.float32),
        np.array([1.0 if t.endswith('000') or t.endswith('00') else 0.0 for t in texts], np.float32),
    ], 1)
    return np.concatenate([feats, ex], 1).astype(np.float32)


def build(split, golds=None):
    dec = pickle.load(open(W + f'\\cache\\dec_{split}.pkl', 'rb'))
    X, grp, cand, lab = [], [], [], []
    for i, (texts, confs, feats) in enumerate(dec):
        if len(texts) == 0:
            continue
        F = str_feats(texts, confs, feats)
        X.append(F); grp.extend([i] * len(texts)); cand.extend(texts)
        if golds is not None:
            g = golds[i]
            lab.extend([1.0 if t == g else 0.0 for t in texts])
    X = np.concatenate(X) if X else np.zeros((0, 40), np.float32)
    return X, np.array(grp), cand, (np.array(lab, np.float32) if golds is not None else None)


def main():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'] for a in A]
    X, grp, cand, lab = build('train', golds)
    print('candidates', len(X), 'positives', int(lab.sum()))
    hit = np.zeros(len(tr), bool)
    for g, l in zip(grp, lab):
        if l > 0:
            hit[g] = True
    print('recall ceiling (exact string present):', hit.mean())

    gkf = GroupKFold(5)
    oof = np.zeros(len(X), np.float32)
    for ti, vi in gkf.split(X, lab, grp):
        m = lgb.train(dict(objective='binary', learning_rate=0.05, num_leaves=63,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                           min_data_in_leaf=30, verbose=-1, seed=0),
                      lgb.Dataset(X[ti], lab[ti]), num_boost_round=500)
        oof[vi] = m.predict(X[vi])
    # per-image argmax
    preds = {}
    for i in range(len(tr)):
        m = grp == i
        if m.sum() == 0:
            preds[i] = ''
            continue
        idx = np.where(m)[0]
        preds[i] = cand[idx[np.argmax(oof[idx])]]
    scores = [total_price_score(preds[i], golds[i]) for i in range(len(tr))]
    exact = np.mean([preds[i] == golds[i] for i in range(len(tr))])
    print(f'TOTAL_PRICE oof score {np.mean(scores):.4f}  exact {exact:.4f}')
    pickle.dump(preds, open(W + r'\cache\total_oof.pkl', 'wb'))
    return np.mean(scores)


if __name__ == '__main__':
    main()
