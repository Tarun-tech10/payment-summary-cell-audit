"""Train heads on train, predict test, write submission.csv. Reports OOF estimate."""
import os, json, pickle, re, sys
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scorer import total_price_score, grid_score, fields_score, breakdown
from ranker import str_feats, value_of
from numitems import img_feats
from wordfeat import image_wordfeats, token_keyscores, KEYS, F_RELY, F_P2, F_P3, F_RIGHT

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
FKEYS = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
NI_MAX = 15


# ---------------------------------------------------------------- candidates
def cand_features(dec_row):
    """Rank only digit-bearing tokens, but use every token for line context."""
    texts, confs, F = dec_row
    if len(texts) == 0:
        return np.zeros((0, 1), np.float32), []
    keep = np.array([len(re.sub(r'\D', '', t)) >= 2 for t in texts])
    if not keep.any():
        return np.zeros((0, 1), np.float32), []
    ktexts = [t for t, k in zip(texts, keep) if k]
    base = str_feats(ktexts, confs[keep], F[keep])
    S, ks = token_keyscores(texts)
    li = F[:, 8]
    idx = np.where(keep)[0]
    extra = np.zeros((len(idx), 2 * len(ks) + 4), np.float32)
    for j, i in enumerate(idx):
        same = (li == li[i])
        left = same & (F[:, 1] < F[i, 1])
        if same.any():
            extra[j, :len(ks)] = S[same].max(0)
        if left.any():
            extra[j, len(ks):2 * len(ks)] = S[left].max(0)
        extra[j, 2 * len(ks)] = same.sum()
        extra[j, 2 * len(ks) + 1] = left.sum()
        extra[j, 2 * len(ks) + 2] = S[i].max()
        extra[j, 2 * len(ks) + 3] = float(np.argmax(S[i]))
    return np.concatenate([base, extra], 1).astype(np.float32), ktexts


def build_cands(split, n, golds=None):
    dec = pickle.load(open(W + f'\\cache\\dec_{split}.pkl', 'rb'))
    Xs, grp, cands, lab = [], [], [], []
    for i in range(n):
        F, texts = cand_features(dec[i])
        if len(texts) == 0:
            continue
        Xs.append(F); grp.extend([i] * len(texts)); cands.extend(texts)
        if golds is not None:
            lab.extend([1.0 if t.strip() == golds[i].strip() else 0.0 for t in texts])
    X = np.concatenate(Xs) if Xs else np.zeros((0, 1), np.float32)
    return X, np.array(grp), cands, (np.array(lab, np.float32) if golds is not None else None)


def pick(pred, grp, cands, n, fallback='25.000'):
    out = []
    for i in range(n):
        m = np.where(grp == i)[0]
        out.append(cands[m[np.argmax(pred[m])]] if len(m) else fallback)
    return out


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'] for a in A]
    ntr, nte = len(tr), len(te)

    # grid: average of the two CNN runs (best on OOF); counts/fields: the longer run
    og = 0.5 * np.load(W + r'\cache\oof_g.npy') + 0.5 * np.load(W + r'\cache\oof2_g.npy')
    tg = 0.5 * np.load(W + r'\cache\te1_g.npy') + 0.5 * np.load(W + r'\cache\te2_g.npy')
    on = np.load(W + r'\cache\oof2_n.npy'); tn = np.load(W + r'\cache\te2_n.npy')
    of = np.load(W + r'\cache\oof2_f.npy'); tf = np.load(W + r'\cache\te2_f.npy')

    # ---------------- total_price
    Xtr, gtr, ctr, ltr = build_cands('train', ntr, golds)
    Xte, gte, cte, _ = build_cands('test', nte)
    print('cand train', Xtr.shape, 'pos', int(ltr.sum()))
    hit = np.zeros(ntr, bool)
    for g, l in zip(gtr, ltr):
        if l > 0:
            hit[g] = True
    print('recall ceiling', hit.mean())
    P = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
    oofp = np.zeros(len(Xtr), np.float32)
    for ti, vi in GroupKFold(5).split(Xtr, ltr, gtr):
        m = lgb.train(P, lgb.Dataset(Xtr[ti], ltr[ti]), num_boost_round=500)
        oofp[vi] = m.predict(Xtr[vi])
    tot_oof = pick(oofp, gtr, ctr, ntr)
    ts = np.mean([total_price_score(tot_oof[i], golds[i]) for i in range(ntr)])
    print(f'TOTAL oof {ts:.4f} exact {np.mean([tot_oof[i]==golds[i] for i in range(ntr)]):.4f}')
    mfull = lgb.train(P, lgb.Dataset(Xtr, ltr), num_boost_round=500)
    tot_te = pick(mfull.predict(Xte), gte, cte, nte)

    # ---------------- num_items
    dtr = pickle.load(open(W + r'\cache\dec_train.pkl', 'rb'))
    dte = pickle.load(open(W + r'\cache\dec_test.pkl', 'rb'))
    Ltr = np.stack([np.concatenate([img_feats(dtr[i]), image_wordfeats(dtr[i])]) for i in range(ntr)])
    Lte = np.stack([np.concatenate([img_feats(dte[i]), image_wordfeats(dte[i])]) for i in range(nte)])
    y_ni = np.array([min(a['num_items'], NI_MAX) for a in A])
    Xn = np.concatenate([Ltr, on], 1); Xnte = np.concatenate([Lte, tn], 1)
    PN = dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=15, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=0)
    oni = np.zeros((ntr, NI_MAX + 1), np.float32)
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, np.minimum(y_ni, 8)):
        m = lgb.train(PN, lgb.Dataset(Xn[ti], y_ni[ti]), num_boost_round=250)
        oni[vi] = m.predict(Xn[vi])
    accs = {w: ((w * oni + (1 - w) * on).argmax(1) == y_ni).mean() for w in [0, .25, .5, .75, 1]}
    bw = max(accs, key=accs.get)
    print('num_items acc by blend', {k: round(v, 4) for k, v in accs.items()}, 'best', bw)
    ni_oof = (bw * oni + (1 - bw) * on).argmax(1)
    mfull = lgb.train(PN, lgb.Dataset(Xn, y_ni), num_boost_round=250)
    ni_te = (bw * mfull.predict(Xnte) + (1 - bw) * tn).argmax(1)

    # ---------------- fields
    y_f = np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32)
    Xf = np.concatenate([Ltr, of], 1); Xfte = np.concatenate([Lte, tf], 1)
    PF = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    off = np.zeros((ntr, 4), np.float32); tfp = np.zeros((nte, 4), np.float32)
    for j in range(4):
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y_f[:, j]):
            m = lgb.train(PF, lgb.Dataset(Xf[ti], y_f[ti, j]), num_boost_round=250)
            off[vi, j] = m.predict(Xf[vi])
        m = lgb.train(PF, lgb.Dataset(Xf, y_f[:, j]), num_boost_round=250)
        tfp[:, j] = m.predict(Xfte)
    def fsc(p):
        return np.mean([fields_score(dict(zip(FKEYS, p[i] > .5)), dict(zip(FKEYS, y_f[i] > .5)))
                        for i in range(ntr)])
    fs_cnn, fs_lgb = fsc(of), fsc(off)
    ws = {w: fsc(w * off + (1 - w) * of) for w in [0, .25, .5, .75, 1]}
    bwf = max(ws, key=ws.get)
    print('fields cnn', round(fs_cnn, 4), 'lgb', round(fs_lgb, 4), 'blend', {k: round(v, 4) for k, v in ws.items()})
    f_oof = (bwf * off + (1 - bwf) * of) > 0.5
    f_te = (bwf * tfp + (1 - bwf) * tf) > 0.5

    # ---------------- grid
    G = np.array([a['semantic_grid'] for a in A])
    g_oof = og.argmax(1).reshape(ntr, 16)
    g_te = tg.argmax(1).reshape(nte, 16)
    gs = np.mean([grid_score(g_oof[i], G[i]) for i in range(ntr)])
    print('grid oof', round(gs, 4))

    preds_oof = [dict(semantic_grid=g_oof[i].tolist(), total_price=tot_oof[i],
                      num_items=int(ni_oof[i]), fields_present=dict(zip(FKEYS, [bool(x) for x in f_oof[i]])))
                 for i in range(ntr)]
    print('OOF BREAKDOWN', {k: round(v, 4) for k, v in breakdown(preds_oof, A).items()})

    rows = []
    for i in range(nte):
        rows.append(dict(id=te.id[i], answer_json=json.dumps(dict(
            fields_present={k: bool(v) for k, v in zip(FKEYS, f_te[i])},
            num_items=int(ni_te[i]),
            semantic_grid=[int(x) for x in g_te[i]],
            total_price=tot_te[i]), sort_keys=True)))
    sub = pd.DataFrame(rows)
    sub.to_csv(W + r'\submission.csv', index=False)
    print('wrote submission', sub.shape)


if __name__ == '__main__':
    main()
