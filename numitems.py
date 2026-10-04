"""num_items model: CNN probabilities + layout/OCR line-structure features."""
import os, json, pickle, re
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
NI_MAX = 15

# token feature column indices (see tokens.py)
F_RELY, F_RELX, F_X1, F_TW, F_LHR, F_NG, F_HM, F_HS, F_LI, F_LIB, F_TI, F_NTOK, \
    F_RIGHT, F_P0, F_P1, F_P2, F_P3, F_YC, F_XC, F_LH, F_NLINES, F_RH, F_RW, F_SW = range(24)


def img_feats(dec_row):
    texts, confs, F = dec_row
    if len(texts) == 0:
        return np.zeros(34, np.float32)
    n = len(texts)
    li = F[:, F_LI] * np.maximum(F[:, F_NLINES] - 1, 1)
    nlines = float(F[0, F_NLINES])
    p1, p2, p3, p0 = F[:, F_P1], F[:, F_P2], F[:, F_P3], F[:, F_P0]
    isnum = np.array([1.0 if re.search(r'\d', t) else 0.0 for t in texts], np.float32)
    vals = np.array([float(re.sub(r'\D', '', t) or -1) for t in texts], np.float32)
    right = F[:, F_RIGHT]
    # lines that own a right-aligned numeric token
    def cnt(mask):
        if mask.sum() == 0:
            return 0.0
        return float(len(np.unique(li[mask])))
    menu = (p1 > 0.35)
    numright = (isnum > 0) & (right > 0)
    out = [
        nlines, cnt(np.ones(n, bool)), cnt(numright), cnt(menu), cnt(menu & numright),
        cnt(p2 > 0.35), cnt(p3 > 0.35), cnt((p3 > 0.35) & numright),
        cnt((p0 > 0.5)), float(n), float(isnum.sum()), float(numright.sum()),
        float(np.median(F[:, F_LH])), float(F[0, F_RH]), float(F[0, F_RW]),
        float(np.median(F[:, F_NG])), float((vals > 0).sum()),
        float(len(set(vals[vals > 0].tolist()))),
        float(np.median(vals[vals > 0]) if (vals > 0).any() else 0),
        float(F[:, F_RELY][numright].mean() if numright.any() else 0),
        float(np.percentile(F[:, F_RELY], 10)), float(np.percentile(F[:, F_RELY], 90)),
        float((F[:, F_NTOK] >= 2).mean()), float(np.median(F[:, F_NTOK])),
        cnt(numright & (F[:, F_RELY] < 0.5)), cnt(numright & (F[:, F_RELY] >= 0.5)),
        float(F[0, F_RH] / max(np.median(F[:, F_LH]) / max(F[0, F_RH], 1e-6), 1e-6)),
        float(np.median(F[:, F_LH]) / max(F[0, F_RH], 1e-6)),
        cnt(menu & (right > 0)), float(menu.sum()), float((p2 > 0.35).sum()),
        float((p3 > 0.35).sum()), float(np.mean(p1)), float(np.mean(p3)),
    ]
    return np.array(out, np.float32)


def build(split, n):
    dec = pickle.load(open(W + f'\\cache\\dec_{split}.pkl', 'rb'))
    return np.stack([img_feats(dec[i]) for i in range(n)])


def main():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    y = np.array([min(a['num_items'], NI_MAX) for a in A])
    cnnp = np.load(W + r'\cache\oof2_n.npy') if os.path.exists(W + r'\cache\oof2_n.npy') \
        else np.load(W + r'\cache\oof_n.npy')
    Xl = build('train', len(tr))
    X = np.concatenate([Xl, cnnp], 1)
    print('cnn alone acc', (cnnp.argmax(1) == y).mean())
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    oof = np.zeros((len(y), NI_MAX + 1), np.float32)
    for ti, vi in skf.split(X, np.minimum(y, 8)):
        m = lgb.train(dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=0.05,
                           num_leaves=15, min_data_in_leaf=15, feature_fraction=0.7,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0),
                      lgb.Dataset(X[ti], y[ti]), num_boost_round=250)
        oof[vi] = m.predict(X[vi])
    print('lgbm+cnn acc', (oof.argmax(1) == y).mean())
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bl = w * oof + (1 - w) * cnnp
        print(' blend w', w, (bl.argmax(1) == y).mean())
    np.save(W + r'\cache\ni_oof_lgb.npy', oof)


if __name__ == '__main__':
    main()
