"""Final assembly: total_price ranker + num_items + fields + grid -> submission.csv"""
import os, json, pickle, re, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold, StratifiedKFold
from scorer import total_price_score, grid_score, fields_score, breakdown
from wordfeat import token_keyscores, KEYS

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
FKEYS = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
NI_MAX = 15
NUM = re.compile(r'-?\d[\d.,]*\d|\d')
KL = list(KEYS)

# tokens2 feature columns
C_RELY, C_RELX, C_X1, C_RELW, C_LHR, C_NG, C_KIND, C_LI, C_LIB, C_TI, C_NTOK, C_RIGHT, \
    C_P0, C_P1, C_P2, C_P3, C_YC, C_XC, C_LH, C_NLINES, C_RH, C_RW, C_SW, C_X0, C_ANG, C_CH = range(26)


def dg(s):
    return re.sub(r'\D', '', s)


def cands_for_image(row):
    """-> (feature matrix, candidate strings)"""
    texts, confs, F = row
    if len(texts) == 0:
        return np.zeros((0, 1), np.float32), []
    n = len(texts)
    li = np.round(F[:, C_LI] * max(F[0, C_NLINES] - 1, 1)).astype(int)
    S, _ = token_keyscores(texts)
    # per-line keyword profile (max over that line's strips)
    lk = defaultdict(lambda: np.zeros(len(KL), np.float32))
    for j in range(n):
        lk[li[j]] = np.maximum(lk[li[j]], S[j])
    rows, strs = [], []
    all_vals = []
    for j in range(n):
        t = texts[j].strip()
        opts = []
        if t and dg(t):
            opts.append((t, 1.0, 0, 1))
        subs = NUM.findall(t)
        for k, m in enumerate(subs):
            m = m.strip('.,')
            if m and dg(m):
                opts.append((m, 0.0, k, 1.0 if k == len(subs) - 1 else 0.0))
        for (s, isfull, kidx, islast) in opts:
            if len(dg(s)) < 2 or len(s) > 20:
                continue
            rows.append((j, s, isfull, kidx, islast, len(subs)))
            all_vals.append(float(dg(s)))
    if not rows:
        return np.zeros((0, 1), np.float32), []
    vals = np.array(all_vals, np.float32)
    mx = max(vals.max(), 1.0)
    scnt = Counter(s for _, s, _, _, _, _ in rows)
    dcnt = Counter(dg(s) for _, s, _, _, _, _ in rows)
    rank = np.argsort(np.argsort(-vals)).astype(np.float32)
    X, out = [], []
    for r, (j, s, isfull, kidx, islast, nsub) in enumerate(rows):
        d = dg(s)
        sep = ',' if ',' in s else ('.' if '.' in s else '')
        f = list(F[j]) + [
            float(confs[j]), len(s), len(d), 1.0 if sep == ',' else 0.0,
            1.0 if sep == '.' else 0.0, 1.0 if sep == '' else 0.0,
            1.0 if s.startswith('-') else 0.0,
            1.0 if re.search(r'[A-Za-z]', texts[j]) else 0.0,
            isfull, kidx, islast, nsub,
            vals[r] / mx, float(np.log1p(max(vals[r], 0))), rank[r], rank[r] / max(len(rows) - 1, 1),
            1.0 if vals[r] == vals.max() else 0.0,
            scnt[s], dcnt[d], len(rows),
            len(d) - len(d.rstrip('0')),
            1.0 if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+([.,]\d{2})?', s) else 0.0,
            1.0 if re.fullmatch(r'\d+', s) else 0.0,
        ] + list(lk[li[j]]) + list(S[j])
        X.append(f); out.append(s)
    return np.array(X, np.float32), out


def build(split, n, golds=None):
    """Candidates from both reader checkpoints, with a cross-model agreement flag."""
    decs = [pickle.load(open(W + f'\\cache\\dec2_{split}.pkl', 'rb')),
            pickle.load(open(W + f'\\cache\\dec2B_{split}.pkl', 'rb'))]
    Xs, grp, cands, lab = [], [], [], []
    for i in range(n):
        per = [cands_for_image(d[i]) for d in decs]
        sets = [set(cs) for _, cs in per]
        for mi, (F, cs) in enumerate(per):
            if len(cs) == 0:
                continue
            other = sets[1 - mi]
            ag = np.array([[1.0 if c in other else 0.0, float(mi)] for c in cs], np.float32)
            Xs.append(np.concatenate([F, ag], 1))
            grp.extend([i] * len(cs)); cands.extend(cs)
            if golds is not None:
                lab.extend([1.0 if c == golds[i] else 0.0 for c in cs])
    X = np.concatenate(Xs)
    return X, np.array(grp), cands, (np.array(lab, np.float32) if golds is not None else None)


def pick(score, grp, cands, n, fb='25.000'):
    out = []
    for i in range(n):
        m = np.where(grp == i)[0]
        if len(m) == 0:
            out.append(fb); continue
        agg = defaultdict(float)
        for k in m:
            agg[cands[k]] = max(agg[cands[k]], score[k])
        out.append(max(agg, key=agg.get))
    return out


def img_level_feats(dec_row, ext=False):
    texts, confs, F = dec_row
    if len(texts) == 0:
        return np.zeros(20 + 2 * len(KL), np.float32)
    S, _ = token_keyscores(texts)
    nl = float(F[0, C_NLINES])
    isnum = np.array([1.0 if re.search(r'\d\d', t) else 0.0 for t in texts], np.float32)
    li = np.round(F[:, C_LI] * max(nl - 1, 1)).astype(int)
    tokk = F[:, C_KIND] > 0

    def nline(mask):
        return float(len(np.unique(li[mask]))) if mask.any() else 0.0
    p1 = F[:, C_P1] > 0.35; p2 = F[:, C_P2] > 0.35; p3 = F[:, C_P3] > 0.35
    right = F[:, C_RIGHT] > 0
    base = [nl, float(len(texts)), float(isnum.sum()), nline(isnum > 0),
            nline(p1), nline(p2), nline(p3), nline(p1 & (isnum > 0) & right & tokk),
            nline((isnum > 0) & right & tokk), nline(p3 & (isnum > 0)),
            float(np.median(F[:, C_LH])), float(F[0, C_RH]), float(F[0, C_RW]),
            float(np.mean(confs)), float(F[0, C_ANG]), float(F[0, C_CH]),
            float(np.mean(F[:, C_P1])), float(np.mean(F[:, C_P2])),
            float(np.mean(F[:, C_P3])), float(np.mean(F[:, C_P0]))]
    kmax = S.max(0)
    kcnt = (S > 0.85).sum(0).astype(np.float32)
    # many receipts print the item count directly ("ITEMS: 3", "Total Item 4")
    ik = KL.index('item')
    itemlines = set(li[S[:, ik] > 0.8].tolist())
    small = []
    for j in range(len(texts)):
        if li[j] not in itemlines:
            continue
        for m in NUM.findall(texts[j]):
            d = re.sub(r'\D', '', m)
            if 1 <= len(d) <= 2:
                small.append(int(d))
    small = [v for v in small if 0 <= v <= 60]
    cnt = Counter(small)
    top = cnt.most_common(2)
    extra = [float(len(itemlines)), float(len(small)),
             float(top[0][0]) if top else -1.0, float(top[0][1]) if top else 0.0,
             float(top[1][0]) if len(top) > 1 else -1.0,
             float(min(small)) if small else -1.0, float(max(small)) if small else -1.0]
    if not ext:
        return np.array(base + list(kmax) + list(kcnt), np.float32)
    return np.array(base + list(kmax) + list(kcnt) + extra, np.float32)


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    ntr, nte = len(tr), len(te)

    # three layout runs; grid weights picked on out-of-fold grid score
    og = (2 * np.load(W + r'\cache\oof_g.npy') + np.load(W + r'\cache\oof2_g.npy')
          + np.load(W + r'\cache\oof3_g.npy')) / 4
    tg = (2 * np.load(W + r'\cache\te1_g.npy') + np.load(W + r'\cache\te2_g.npy')
          + np.load(W + r'\cache\te3_g.npy')) / 4
    on = np.load(W + r'\cache\oof2_n.npy'); tn = np.load(W + r'\cache\te2_n.npy')
    of = np.load(W + r'\cache\oof2_f.npy'); tf = np.load(W + r'\cache\te2_f.npy')

    # ---- total_price
    Xtr, gtr, ctr, ltr = build('train', ntr, golds)
    Xte, gte, cte, _ = build('test', nte)
    hit = np.zeros(ntr, bool)
    for g, l in zip(gtr, ltr):
        if l > 0:
            hit[g] = True
    print('cands', Xtr.shape, 'pos', int(ltr.sum()), 'recall ceiling', round(hit.mean(), 4), flush=True)
    P = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.75,
             bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40, verbose=-1, seed=0)
    oofp = np.zeros(len(Xtr), np.float32)
    for ti, vi in GroupKFold(5).split(Xtr, ltr, gtr):
        m = lgb.train(P, lgb.Dataset(Xtr[ti], ltr[ti]), num_boost_round=600)
        oofp[vi] = m.predict(Xtr[vi])
    tot_oof = pick(oofp, gtr, ctr, ntr)
    ts = np.mean([total_price_score(tot_oof[i], golds[i]) for i in range(ntr)])
    ex = np.mean([tot_oof[i] == golds[i] for i in range(ntr)])
    print(f'TOTAL oof {ts:.4f}  exact {ex:.4f}', flush=True)
    mf = lgb.train(P, lgb.Dataset(Xtr, ltr), num_boost_round=600)
    tot_te = pick(mf.predict(Xte), gte, cte, nte)

    # ---- image-level features
    dtr = pickle.load(open(W + r'\cache\dec2_train.pkl', 'rb'))
    dte = pickle.load(open(W + r'\cache\dec2_test.pkl', 'rb'))
    Ltr = np.stack([img_level_feats(dtr[i]) for i in range(ntr)])
    Lte = np.stack([img_level_feats(dte[i]) for i in range(nte)])
    Etr = np.stack([img_level_feats(dtr[i], True) for i in range(ntr)])
    Ete = np.stack([img_level_feats(dte[i], True) for i in range(nte)])

    # ---- num_items
    y_ni = np.array([min(a['num_items'], NI_MAX) for a in A])
    Xn = np.concatenate([Ltr, on], 1); Xnte = np.concatenate([Lte, tn], 1)
    PN = dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=15, feature_fraction=0.7, bagging_fraction=0.8,
              bagging_freq=1, verbose=-1, seed=0)
    oni = np.zeros((ntr, NI_MAX + 1), np.float32)
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, np.minimum(y_ni, 8)):
        m = lgb.train(PN, lgb.Dataset(Xn[ti], y_ni[ti]), num_boost_round=300)
        oni[vi] = m.predict(Xn[vi])
    accs = {w: float(((w * oni + (1 - w) * on).argmax(1) == y_ni).mean()) for w in [0, .25, .5, .75, 1]}
    bw = max(accs, key=accs.get)
    print('num_items', {k: round(v, 4) for k, v in accs.items()}, 'best w', bw, flush=True)
    ni_oof = (bw * oni + (1 - bw) * on).argmax(1)
    mf = lgb.train(PN, lgb.Dataset(Xn, y_ni), num_boost_round=300)
    ni_te = (bw * mf.predict(Xnte) + (1 - bw) * tn).argmax(1)

    # ---- fields
    y_f = np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32)
    Xf = np.concatenate([Etr, of], 1); Xfte = np.concatenate([Ete, tf], 1)
    PF = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    off = np.zeros((ntr, 4), np.float32); tfp = np.zeros((nte, 4), np.float32)
    for j in range(4):
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y_f[:, j]):
            m = lgb.train(PF, lgb.Dataset(Xf[ti], y_f[ti, j]), num_boost_round=300)
            off[vi, j] = m.predict(Xf[vi])
        tfp[:, j] = lgb.train(PF, lgb.Dataset(Xf, y_f[:, j]), num_boost_round=300).predict(Xfte)

    def fsc(p):
        return float(np.mean([fields_score(dict(zip(FKEYS, p[i] > .5)),
                                           dict(zip(FKEYS, y_f[i] > .5))) for i in range(ntr)]))
    ws = {w: fsc(w * off + (1 - w) * of) for w in [0, .25, .5, .75, 1]}
    bwf = max(ws, key=ws.get)
    print('fields', {k: round(v, 4) for k, v in ws.items()}, 'best w', bwf, flush=True)
    f_oof = (bwf * off + (1 - bwf) * of) > 0.5
    f_te = (bwf * tfp + (1 - bwf) * tf) > 0.5

    # ---- grid
    G = np.array([a['semantic_grid'] for a in A])
    g_oof = og.argmax(1).reshape(ntr, 16); g_te = tg.argmax(1).reshape(nte, 16)
    print('grid oof', round(float(np.mean([grid_score(g_oof[i], G[i]) for i in range(ntr)])), 4))

    preds = [dict(semantic_grid=g_oof[i].tolist(), total_price=tot_oof[i],
                  num_items=int(ni_oof[i]),
                  fields_present=dict(zip(FKEYS, [bool(x) for x in f_oof[i]]))) for i in range(ntr)]
    bd = breakdown(preds, A)
    print('OOF BREAKDOWN', {k: round(v, 4) for k, v in bd.items()}, flush=True)

    rows = [dict(id=te.id[i], answer_json=json.dumps(dict(
        fields_present={k: bool(v) for k, v in zip(FKEYS, f_te[i])},
        num_items=int(ni_te[i]), semantic_grid=[int(x) for x in g_te[i]],
        total_price=tot_te[i]), sort_keys=True)) for i in range(nte)]
    pd.DataFrame(rows).to_csv(W + r'\submission.csv', index=False)
    print('wrote submission.csv', len(rows))


if __name__ == '__main__':
    main()
