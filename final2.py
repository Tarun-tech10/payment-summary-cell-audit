"""Assembly v2: fold-honest reader decodes + optional beam candidates + ensembled grid."""
import os, json, pickle, re, sys, itertools
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
C_RELY, C_RELX, C_X1, C_RELW, C_LHR, C_NG, C_KIND, C_LI, C_LIB, C_TI, C_NTOK, C_RIGHT, \
    C_P0, C_P1, C_P2, C_P3, C_YC, C_XC, C_LH, C_NLINES, C_RH, C_RW, C_SW, C_X0, C_ANG, C_CH = range(26)


def dg(s):
    return re.sub(r'\D', '', s)


def group3(d, sep):
    if len(d) <= 3:
        return d
    out = []
    while len(d) > 3:
        out.append(d[-3:]); d = d[:-3]
    out.append(d)
    return sep.join(reversed(out))


def reformats(d, sep_style):
    """Plausible printed forms of a digit string (grouping is deterministic)."""
    v = {d, group3(d, ','), group3(d, '.')}
    if len(d) > 4:
        head, tail = d[:-2], d[-2:]
        for s in (',', '.'):
            v.add(group3(head, ',') + s + tail)
            v.add(group3(head, '.') + s + tail)
    if sep_style:
        v.add(group3(d, sep_style))
    return {x for x in v if x}


# Letter/digit lookalike repair (O->0, S->5, ...) was tried here and recovered nothing:
# recall was identical to 4 decimal places, because the reader already emits digits where
# digits belong.  The 31% of receipts with no correct candidate are genuine misreads.


def receipt_sep(texts):
    c = Counter()
    for t in texts:
        for m in NUM.findall(t):
            if len(dg(m)) >= 4:
                for ch in m:
                    if ch in ',.':
                        c[ch] += 1
    return c.most_common(1)[0][0] if c else ''


# Restricting candidates to the summary block cost 1.9 points of total_price the first time
# it was tried -- but only because the filter was applied BEFORE the pool statistics were
# computed, and the duplicate-support / value-rank features are defined over the whole
# receipt.  Building the statistics over every strip and filtering only at emission keeps
# those features intact while removing distractors the ranker can never need.
SUMMARY_MIN = float(os.environ.get('SUMMARY_MIN', -1.0))


def summary_strips(F):
    return (F[:, C_P2] + F[:, C_P3]) >= SUMMARY_MIN


def tta_shape(texts, F):
    """decode_tta stacks S scales by tiling the feature block, so the block repeats."""
    m = len(texts)
    if m % 3 == 0 and m >= 3:
        q = m // 3
        if np.array_equal(F[:q], F[q:2 * q]):
            return 3, q
    return 1, m


def cands_for_image(row, beam=None):
    """row = (texts, confs, F); beam = [(strip_idx, [(str, score), ...]), ...] or None.

    Readings of the SAME strip at different width scales collapse to one candidate row.
    Emitting one row per scale triples the pool and wrecks the duplicate-support and
    value-rank features, which are computed over it; how many scales agreed is far more
    useful as a feature than as three near-identical rows.
    """
    texts, confs, F = row
    if len(texts) == 0:
        return np.zeros((0, 1), np.float32), []
    ns, n = tta_shape(texts, F)
    keep = summary_strips(F[:n])
    li = np.round(F[:n, C_LI] * max(F[0, C_NLINES] - 1, 1)).astype(int)
    S, _ = token_keyscores(texts)
    Sj = np.stack([S[k * n:(k + 1) * n] for k in range(ns)]).max(0)
    lk = defaultdict(lambda: np.zeros(len(KL), np.float32))
    for j in range(n):
        lk[li[j]] = np.maximum(lk[li[j]], Sj[j])
    rsep = receipt_sep(texts)
    bmap = defaultdict(list)
    if beam:
        for j, cl in beam:
            for r, (s, sc) in enumerate(cl):
                bmap[j].append((s, sc, r))
    rows, vals = [], []
    for j in range(n):
        agg = {}                       # (string, brank) -> [isfull, kidx, islast, nsub, bsc, nscale, maxconf, sumconf]
        for k in range(ns):
            t = texts[j + k * n].strip()
            cf = float(confs[j + k * n])
            subs = NUM.findall(t)
            opts = []
            if t and dg(t):
                opts.append((t, 1.0, 0, 1.0, 0.0, -1))
            for q, m in enumerate(subs):
                m = m.strip('.,')
                if m and dg(m):
                    opts.append((m, 0.0, q, 1.0 if q == len(subs) - 1 else 0.0, 0.0, -1))
            # canonical re-formats: if the digits are right, grouping recovers the printed form
            extra = []
            for (s, _, _, _, _, _) in opts:
                d = dg(s)
                if 2 <= len(d) <= 9:
                    for v in reformats(d, rsep):
                        if v != s:
                            extra.append((v, 0.0, 0, 1.0, 0.0, -2))
            if k == 0:
                for (s, sc, r) in bmap.get(j, []):
                    s = s.strip('.,')
                    extra.append((s, 0.0, 0, 1.0, float(sc), r))
                    # one canonical regrouping only -- the full reformat fan-out would
                    # multiply the alternate readings by ~5 and swamp the ranker
                    d = dg(s)
                    if 2 <= len(d) <= 9:
                        c = group3(d, rsep or ',')
                        if c != s:
                            extra.append((c, 0.0, 0, 1.0, float(sc), r))
            for (s, isfull, kidx, islast, bsc, brank) in opts + extra:
                if len(dg(s)) < 2 or len(s) > 20:
                    continue
                key = (s, brank)
                if key in agg:
                    a = agg[key]
                    a[0] = max(a[0], isfull); a[3] = max(a[3], float(len(subs)))
                    a[5] += 1; a[6] = max(a[6], cf); a[7] += cf
                else:
                    agg[key] = [isfull, kidx, islast, float(len(subs)), bsc, 1, cf, cf]
        for (s, brank), a in agg.items():
            rows.append((j, s, a[0], a[1], a[2], a[3], a[4], brank,
                         a[5], a[6], a[7] / max(a[5], 1), a[5] / float(ns)))
            vals.append(float(dg(s)))
    if not rows:
        return np.zeros((0, 1), np.float32), []
    vals = np.array(vals, np.float32); mx = max(vals.max(), 1.0)
    scnt = Counter(r[1] for r in rows); dcnt = Counter(dg(r[1]) for r in rows)
    rank = np.argsort(np.argsort(-vals)).astype(np.float32)
    X, out = [], []
    for r, (j, s, isfull, kidx, islast, nsub, bsc, brank,
            nscale, maxconf, meanconf, scfrac) in enumerate(rows):
        if not keep[j]:                # statistics above were built over the whole receipt
            continue
        d = dg(s)
        sep = ',' if ',' in s else ('.' if '.' in s else '')
        X.append(list(F[j]) + [
            maxconf, len(s), len(d), 1. if sep == ',' else 0., 1. if sep == '.' else 0.,
            1. if sep == '' else 0., 1. if s.startswith('-') else 0.,
            1. if re.search(r'[A-Za-z]', texts[j]) else 0., isfull, kidx, islast, nsub,
            vals[r] / mx, float(np.log1p(max(vals[r], 0))), rank[r], rank[r] / max(len(rows) - 1, 1),
            1. if vals[r] == vals.max() else 0., scnt[s], dcnt[d], len(rows),
            len(d) - len(d.rstrip('0')),
            1. if re.fullmatch(r'-?\d{1,3}([.,]\d{3})+([.,]\d{2})?', s) else 0.,
            1. if re.fullmatch(r'\d+', s) else 0.,
            bsc, float(brank), 1. if brank >= 0 else 0.,
            1. if brank == -2 else 0.,                       # is a canonical re-format
            1. if (sep and sep == rsep) else 0.,             # agrees with the receipt's style
            1. if s == group3(d, rsep or ',') else 0.,
            # how many width scales produced this exact reading -- agreement is evidence
            float(nscale), meanconf, scfrac, maxconf - meanconf,
        ] + list(lk[li[j]]) + list(Sj[j]))
        out.append(s)
    return np.array(X, np.float32), out


def img_level_feats(row, ext=False):
    texts, confs, F = row
    if len(texts) == 0:
        return np.zeros(20 + 2 * len(KL) + (7 if ext else 0), np.float32)
    S, _ = token_keyscores(texts)
    nl = float(F[0, C_NLINES])
    isnum = np.array([1. if re.search(r'\d\d', t) else 0. for t in texts], np.float32)
    li = np.round(F[:, C_LI] * max(nl - 1, 1)).astype(int)
    tokk = F[:, C_KIND] > 0

    def nline(m):
        return float(len(np.unique(li[m]))) if m.any() else 0.
    p1 = F[:, C_P1] > .35; p2 = F[:, C_P2] > .35; p3 = F[:, C_P3] > .35
    right = F[:, C_RIGHT] > 0
    base = [nl, float(len(texts)), float(isnum.sum()), nline(isnum > 0), nline(p1), nline(p2),
            nline(p3), nline(p1 & (isnum > 0) & right & tokk), nline((isnum > 0) & right & tokk),
            nline(p3 & (isnum > 0)), float(np.median(F[:, C_LH])), float(F[0, C_RH]),
            float(F[0, C_RW]), float(np.mean(confs)), float(F[0, C_ANG]), float(F[0, C_CH]),
            float(np.mean(F[:, C_P1])), float(np.mean(F[:, C_P2])), float(np.mean(F[:, C_P3])),
            float(np.mean(F[:, C_P0]))]
    out = base + list(S.max(0)) + list((S > .85).sum(0).astype(np.float32))
    if ext:
        ik = KL.index('item')
        ilines = set(li[S[:, ik] > .8].tolist())
        small = [int(dg(m)) for j in range(len(texts)) if li[j] in ilines
                 for m in NUM.findall(texts[j]) if 1 <= len(dg(m)) <= 2]
        small = [v for v in small if 0 <= v <= 60]
        top = Counter(small).most_common(2)
        out += [float(len(ilines)), float(len(small)),
                float(top[0][0]) if top else -1., float(top[0][1]) if top else 0.,
                float(top[1][0]) if len(top) > 1 else -1.,
                float(min(small)) if small else -1., float(max(small)) if small else -1.]
    return np.array(out, np.float32)


def load_dec(split, use_beam):
    """Prefer the fold-honest decodes; fall back to the earlier caches."""
    # Width-scale TTA (dec6/dec7) raises candidate RECALL 0.6465 -> 0.6749 but costs more
    # than that in selection: it multiplies the distractors per receipt by ~2.4x, and
    # total_price fell 0.5959 -> 0.5713 (and 0.5596 after de-duplicating).  The extra
    # readings are mostly wrong readings.  Single-scale decodes are the better trade;
    # the TTA packs are still used for the grid refiner's OCR evidence, where a bigger
    # pool of readings costs nothing.
    order = ['dec4', 'dec5', 'dec2', 'dec2B']
    # a decode is only usable if BOTH splits have it -- a half-written cache would
    # silently give train and test different feature matrices
    names = [nm for nm in order
             if os.path.exists(W + f'\\cache\\{nm}_train.pkl')
             and os.path.exists(W + f'\\cache\\{nm}_test.pkl')][:2]
    packs = [pickle.load(open(W + f'\\cache\\{nm}_{split}.pkl', 'rb')) for nm in names]
    beams = None
    if use_beam and os.path.exists(W + f'\\cache\\beamB_{split}.pkl'):
        beams = pickle.load(open(W + f'\\cache\\beamB_{split}.pkl', 'rb'))
    return packs, beams


def build(split, n, golds=None, use_beam=True):
    packs, beams = load_dec(split, use_beam)
    Xs, grp, cands, lab = [], [], [], []
    for i in range(n):
        per = []
        for mi, P in enumerate(packs):
            bm = beams[i] if (beams is not None and mi == len(packs) - 1) else None
            per.append(cands_for_image(P[i], bm))
        sets = [set(cs) for _, cs in per]
        for mi, (F, cs) in enumerate(per):
            if len(cs) == 0:
                continue
            other = set().union(*[s for k, s in enumerate(sets) if k != mi])
            ag = np.array([[1. if c in other else 0., float(mi)] for c in cs], np.float32)
            Xs.append(np.concatenate([F, ag], 1))
            grp.extend([i] * len(cs)); cands.extend(cs)
            if golds is not None:
                lab.extend([1. if c == golds[i] else 0. for c in cs])
    X = np.concatenate(Xs)
    return X, np.array(grp), cands, (np.array(lab, np.float32) if golds is not None else None)


def pattern_of(s):
    return ''.join('D' if c.isdigit() else c for c in s)


def arith_feats(keys):
    """A receipt is a self-consistent ledger: the total is the value that equals
    subtotal+tax+service, and the one cash-minus-change lands on.  These relations hold
    between candidate VALUES, so they are checkable with no labels at all."""
    byimg = defaultdict(set)
    for (g, d) in keys:
        if 1 <= len(d) <= 9:
            byimg[g].add(int(d))
    out = []
    for (g, d) in keys:
        S = byimg[g]
        V = int(d) if 1 <= len(d) <= 9 else -1
        srt = sorted(S)
        mx = srt[-1] if srt else 0
        pairs = taxlike = diffs = halves = 0
        if V > 0:
            for a in srt:
                if a >= V:
                    break
                b = V - a
                if b in S and b > 0:
                    pairs += 1
                    if b <= 0.30 * a:
                        taxlike += 1
            for a in srt:
                if a > V and (a - V) in S:
                    diffs += 1
            halves = 1 if (V % 2 == 0 and V // 2 in S) else 0
        out.append([pairs, taxlike, diffs, halves, sum(1 for a in S if a > V), len(S),
                    V / max(mx, 1), 1.0 * (V == mx),
                    1.0 * (V > 0 and V % 1000 == 0), 1.0 * (V > 0 and V % 100 == 0),
                    1.0 * (V > 0 and V % 500 == 0),
                    float(np.log1p(max(V, 0))), float(np.log1p(max(mx - V, 0)))])
    return np.array(out, np.float32)


def digit_matrix(X, grp, cands, use_arith=True):
    idx_by = defaultdict(list)
    for r, (g, c) in enumerate(zip(grp, cands)):
        idx_by[(g, dg(c))].append(r)
    rows, owner, keys = [], [], []
    for (g, d), rs in idx_by.items():
        sub = X[rs]
        val = float(d) if len(d) <= 12 else 0.0
        rows.append(np.concatenate([sub.max(0), sub.mean(0),
                                    [len(rs), len(d), val, np.log1p(val)]]))
        owner.append(g); keys.append((g, d))
    R = np.array(rows, np.float32)
    if use_arith:
        R = np.concatenate([R, arith_feats(keys)], 1)
    return R, np.array(owner), keys


def select_total(Xtr, gtr, ctr, ltr, Xte, gte, cte, ntr, nte, golds):
    """Stage A picks the digit string, stage B picks how it was printed."""
    P = dict(objective='binary', learning_rate=0.05, num_leaves=63, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=30, verbose=-1, seed=0)
    PA = dict(P, num_leaves=127, min_data_in_leaf=15, learning_rate=0.035)   # tuned on OOF
    XD, ownD, keys = digit_matrix(Xtr, gtr, ctr)
    labD = np.array([1.0 if d == dg(golds[g]) else 0.0 for (g, d) in keys], np.float32)
    oofD = np.zeros(len(XD), np.float32)
    for ti, vi in GroupKFold(5).split(XD, labD, ownD):
        oofD[vi] = lgb.train(PA, lgb.Dataset(XD[ti], labD[ti]),
                             num_boost_round=900).predict(XD[vi])
    mD = lgb.train(PA, lgb.Dataset(XD, labD), num_boost_round=900)
    XDe, ownDe, keyse = digit_matrix(Xte, gte, cte)
    predDe = mD.predict(XDe)

    def best_digits(scores, keys, n):
        out = {}
        for r, (g, d) in enumerate(keys):
            if g not in out or scores[r] > out[g][1]:
                out[g] = (d, scores[r])
        return {i: out.get(i, ('', 0.0))[0] for i in range(n)}

    def mbr_digits(scores, keys, n, temp, K=24):
        """A wrong answer still earns partial credit for digits it shares with the gold, so
        the best pick maximises expected score, not probability of being exactly right."""
        by = defaultdict(list)
        for r, (g, d) in enumerate(keys):
            by[g].append((d, float(scores[r])))
        out = {}
        for g, lst in by.items():
            lst = sorted(lst, key=lambda x: -x[1])[:K]
            ds = [d for d, _ in lst]
            w = np.clip(np.array([s for _, s in lst], np.float64), 1e-9, None) ** (1.0 / temp)
            w /= w.sum()
            M = np.array([[total_price_score(a, b) for b in ds] for a in ds])
            out[g] = ds[int((M * w).sum(1).argmax())]
        return {i: out.get(i, '') for i in range(n)}

    bd_arg = best_digits(oofD, keys, ntr)
    dacc = np.mean([bd_arg[i] == dg(golds[i]) for i in range(ntr)])
    scored = [(float(np.mean([total_price_score(bd_arg[i], dg(golds[i])) for i in range(ntr)])),
               'argmax', None)]
    for tp in (0.5, 1.0, 2.0):
        m = mbr_digits(oofD, keys, ntr, tp)
        scored.append((float(np.mean([total_price_score(m[i], dg(golds[i]))
                                      for i in range(ntr)])), 'mbr', tp))
    scored.sort(key=lambda x: -x[0])
    print(f'stage A digit acc {dacc:.4f}; decode ' +
          ', '.join(f'{n_}{t if t else ""}={s:.4f}' for s, n_, t in scored), flush=True)
    # This proxy scores DIGITS only; stage B still has to pick the printed form, so a
    # hairline win here does not survive to the end metric.  Switch only on a real margin.
    argmax_score = [s for s, n_, _ in scored if n_ == 'argmax'][0]
    if scored[0][1] == 'mbr' and scored[0][0] > argmax_score + 0.004:
        tp = scored[0][2]
        bd_tr = mbr_digits(oofD, keys, ntr, tp)
        bd_te = mbr_digits(predDe, keyse, nte, tp)
        print(f'  using MBR digit decode (temp {tp})', flush=True)
    else:
        bd_tr = bd_arg
        bd_te = best_digits(predDe, keyse, nte)

    pri = Counter(pattern_of(g) for g in golds); tot = sum(pri.values())

    def fmt_block(X, grp, cands, bd):
        keep = [r for r in range(len(cands)) if dg(cands[r]) == bd.get(grp[r], '')]
        if not keep:
            return None, None, None, None
        # how this receipt prints numbers of the same digit-length
        recpat = defaultdict(Counter)
        for r in range(len(cands)):
            g = grp[r]
            if len(dg(cands[r])) == len(bd.get(g, '')):
                recpat[g][pattern_of(cands[r])] += 1
        ex = []
        for r in keep:
            g = grp[r]; p = pattern_of(cands[r]); rc = recpat[g]
            modal = rc.most_common(1)[0][0] if rc else ''
            ex.append([np.log((pri.get(p, 0) + 0.5) / (tot + 50.0)),
                       len(cands[r]), 1.0 if cands[r] == dg(cands[r]) else 0.0,
                       float(rc.get(p, 0)), float(rc.get(p, 0)) / max(sum(rc.values()), 1),
                       1.0 if p == modal else 0.0])
        return (np.concatenate([X[keep], np.array(ex, np.float32)], 1),
                grp[keep], [cands[r] for r in keep], keep)
    XB, gB, cB, keep = fmt_block(Xtr, gtr, ctr, bd_tr)
    labB = np.array([1.0 if cB[j] == golds[gB[j]] else 0.0 for j in range(len(cB))], np.float32)
    oofB = np.zeros(len(XB), np.float32)
    for ti, vi in GroupKFold(5).split(XB, labB, gB):
        oofB[vi] = lgb.train(P, lgb.Dataset(XB[ti], labB[ti]),
                             num_boost_round=400).predict(XB[vi])
    mB = lgb.train(P, lgb.Dataset(XB, labB), num_boost_round=400)
    XBe, gBe, cBe, _ = fmt_block(Xte, gte, cte, bd_te)
    predBe = mB.predict(XBe) if XBe is not None else None

    def choose(scores, grps, strs, n):
        out = {}
        for r in range(len(strs)):
            g = grps[r]
            if g not in out or scores[r] > out[g][1]:
                out[g] = (strs[r], scores[r])
        return [out.get(i, ('25.000', 0))[0] for i in range(n)]
    return choose(oofB, gB, cB, ntr), choose(predBe, gBe, cBe, nte)


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


def grid_stack():
    tags = []
    for t in ['', '2', '3', '4', '5']:
        o = W + f'\\cache\\oof{t}_g.npy'
        e = W + (r'\cache\te1_g.npy' if t == '' else f'\\cache\\te{t}_g.npy')
        if os.path.exists(o) and os.path.exists(e):
            tags.append((t, np.load(o), np.load(e)))
    return tags


def main(use_beam=True):
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    ntr, nte = len(tr), len(te)
    G = np.array([a['semantic_grid'] for a in A])

    # ---- grid: the cell-level refiner already blends the CNN runs; else search weights
    if os.path.exists(W + r'\cache\oofC_g.npy'):
        og = np.load(W + r'\cache\oofC_g.npy'); tg = np.load(W + r'\cache\teC_g.npy')
        p = og.argmax(1).reshape(ntr, 16)
        print('grid: cell-refined, oof',
              round(float(np.mean([grid_score(p[i], G[i]) for i in range(ntr)])), 4), flush=True)
    else:
        tags = grid_stack()
        best = None
        for w in itertools.product([0, 1, 2], repeat=len(tags)):
            if sum(w) == 0:
                continue
            P = sum(wi * t[1] for wi, t in zip(w, tags)) / sum(w)
            p = P.argmax(1).reshape(ntr, 16)
            s = float(np.mean([grid_score(p[i], G[i]) for i in range(ntr)]))
            if best is None or s > best[0]:
                best = (s, w)
        gw = best[1]
        og = sum(wi * t[1] for wi, t in zip(gw, tags)) / sum(gw)
        tg = sum(wi * t[2] for wi, t in zip(gw, tags)) / sum(gw)
        print('grid weights', gw, 'oof', round(best[0], 4), flush=True)
    on = np.load(W + r'\cache\oof2_n.npy'); tn = np.load(W + r'\cache\te2_n.npy')
    of = np.load(W + r'\cache\oof2_f.npy'); tf = np.load(W + r'\cache\te2_f.npy')

    # ---- total_price
    Xtr, gtr, ctr, ltr = build('train', ntr, golds, use_beam)
    Xte, gte, cte, _ = build('test', nte, None, use_beam)
    hit = np.zeros(ntr, bool)
    for g, l in zip(gtr, ltr):
        if l > 0:
            hit[g] = True
    print('cands', Xtr.shape, 'recall ceiling', round(hit.mean(), 4), flush=True)
    tot_oof, tot_te = select_total(Xtr, gtr, ctr, ltr, Xte, gte, cte, ntr, nte, golds)
    ts = float(np.mean([total_price_score(tot_oof[i], golds[i]) for i in range(ntr)]))
    ex = float(np.mean([tot_oof[i] == golds[i] for i in range(ntr)]))
    print(f'TOTAL oof {ts:.4f} exact {ex:.4f}', flush=True)

    # ---- image-level features from the honest decodes
    # image-level features from every available reading (different crops/readers see
    # different things, and the flag metric is all-or-nothing so evidence matters)
    def stack_feats(split, n, ext):
        names = [nm for nm in ['dec8', 'dec6', 'dec7', 'dec4', 'dec5', 'dec2', 'dec2B']
                 if os.path.exists(W + f'\\cache\\{nm}_{split}.pkl')][:3]
        mats = []
        for nm in names:
            P = pickle.load(open(W + f'\\cache\\{nm}_{split}.pkl', 'rb'))
            mats.append(np.stack([img_level_feats(P[i], ext) for i in range(n)]))
        return np.concatenate(mats, 1)
    def grid_feats(P):
        """Shape of the predicted map: how much menu / subtotal / total area, and where."""
        out = []
        for i in range(len(P)):
            q = P[i].reshape(4, 16)
            am = q.argmax(0)
            g = am.reshape(4, 4)
            f = [float((am == v).sum()) for v in range(4)]
            f += [float(q[v].sum()) for v in range(4)]
            f += [float(q[v].max()) for v in range(4)]
            for v in (1, 2, 3):
                rs = np.where((g == v).any(1))[0]
                f += [float(rs.min()) if len(rs) else -1.0,
                      float(rs.max()) if len(rs) else -1.0, float(len(rs))]
            cs = np.where((g == 1).any(0))[0]
            f += [float(len(cs))]
            out.append(f)
        return np.array(out, np.float32)
    # grid-shape features were tried here and were neutral-to-slightly-negative; the OCR
    # evidence already carries what they encode.
    Ltr = stack_feats('train', ntr, False); Lte = stack_feats('test', nte, False)
    Etr = stack_feats('train', ntr, True); Ete = stack_feats('test', nte, True)

    y_ni = np.array([min(a['num_items'], NI_MAX) for a in A])
    # an item count is a line count: count the priced lines inside the predicted item band
    try:
        import nifix
        Ntr = nifix.feats('train', ntr, og); Nte = nifix.feats('test', nte, tg)
        Xn = np.concatenate([Ltr, on, Ntr], 1); Xnte = np.concatenate([Lte, tn, Nte], 1)
        print('num_items: + line-count features', Ntr.shape, flush=True)
    except Exception as e:
        print('num_items: line features unavailable', e, flush=True)
        Xn = np.concatenate([Ltr, on], 1); Xnte = np.concatenate([Lte, tn], 1)
    PN = dict(objective='multiclass', num_class=NI_MAX + 1, learning_rate=0.05, num_leaves=15,
              min_data_in_leaf=15, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              verbose=-1, seed=0)
    oni = np.zeros((ntr, NI_MAX + 1), np.float32)
    for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, np.minimum(y_ni, 8)):
        oni[vi] = lgb.train(PN, lgb.Dataset(Xn[ti], y_ni[ti]), num_boost_round=300).predict(Xn[vi])
    accs = {w: float(((w * oni + (1 - w) * on).argmax(1) == y_ni).mean()) for w in [0, .25, .5, .75, 1]}
    bw = max(accs, key=accs.get)
    print('num_items', {k: round(v, 4) for k, v in accs.items()}, 'w', bw, flush=True)
    ni_oof = (bw * oni + (1 - bw) * on).argmax(1)
    ni_te = (bw * lgb.train(PN, lgb.Dataset(Xn, y_ni), num_boost_round=300).predict(Xnte)
             + (1 - bw) * tn).argmax(1)

    y_f = np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32)
    Xf = np.concatenate([Etr, of], 1); Xfte = np.concatenate([Ete, tf], 1)
    PF = dict(objective='binary', learning_rate=0.05, num_leaves=15, min_data_in_leaf=15,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
    off = np.zeros((ntr, 4), np.float32); tfp = np.zeros((nte, 4), np.float32)
    for j in range(4):
        for ti, vi in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, y_f[:, j]):
            off[vi, j] = lgb.train(PF, lgb.Dataset(Xf[ti], y_f[ti, j]),
                                   num_boost_round=300).predict(Xf[vi])
        tfp[:, j] = lgb.train(PF, lgb.Dataset(Xf, y_f[:, j]), num_boost_round=300).predict(Xfte)

    def fsc(p):
        return float(np.mean([fields_score(dict(zip(FKEYS, p[i] > .5)),
                                           dict(zip(FKEYS, y_f[i] > .5))) for i in range(ntr)]))
    ws = {w: fsc(w * off + (1 - w) * of) for w in [0, .25, .5, .75, 1]}
    bwf = max(ws, key=ws.get)
    print('fields', {k: round(v, 4) for k, v in ws.items()}, 'w', bwf, flush=True)

    # fields_score is all-or-nothing (1.0 for four right, 0.25*frac otherwise), so the
    # right target is the most likely JOINT pattern, not four independent thresholds
    FVECS = np.array([[(m >> k) & 1 for k in range(4)] for m in range(16)], np.float32)
    FM = np.array([[1.0 if a == b else 0.25 * float((FVECS[a] == FVECS[b]).mean())
                    for b in range(16)] for a in range(16)], np.float32)

    def fields_decode(p):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(q) @ FVECS.T + np.log(1 - q) @ (1 - FVECS).T
        Pj = np.exp(lp - lp.max(1, keepdims=True)); Pj /= Pj.sum(1, keepdims=True)
        return FVECS[(Pj @ FM.T).argmax(1)] > 0.5

    pf_oof = bwf * off + (1 - bwf) * of
    f_thr = pf_oof > 0.5
    f_joint = fields_decode(pf_oof)

    def fsc2(mask):
        return float(np.mean([fields_score(dict(zip(FKEYS, mask[i])),
                                           dict(zip(FKEYS, y_f[i] > .5))) for i in range(ntr)]))
    a_thr, a_joint = fsc2(f_thr), fsc2(f_joint)
    print(f'fields decode: threshold {a_thr:.4f} vs joint-expected {a_joint:.4f}', flush=True)
    use_joint = a_joint > a_thr
    f_oof = f_joint if use_joint else f_thr
    pf_te = bwf * tfp + (1 - bwf) * tf
    f_te = fields_decode(pf_te) if use_joint else (pf_te > 0.5)

    # ---- grid decode with structural constraints
    def decode(P4, fpred):
        out = []
        for i in range(len(P4)):
            q = P4[i].copy()
            if not fpred[i].any():
                q[2] = -1
            p = q.argmax(0).ravel()
            if not (p == 3).any():
                p[int(np.argmax(q[3].ravel()))] = 3
            out.append(p)
        return np.array(out)

    gate = W + r'\cache\pres_thr.json'
    if os.path.exists(gate):
        # calibrated class biases + an image-level presence gate, both tuned on the metric:
        # macro-IoU charges a full zero term for any class predicted that the gold lacks
        import gridpres
        cfg = json.load(open(gate))
        thr = {int(k): v for k, v in cfg['thr'].items()}
        hi = {int(k): v for k, v in cfg['hi'].items()}
        bias = cfg['bias']
        po = np.load(W + r'\cache\pres_oof.npy'); pt = np.load(W + r'\cache\pres_te.npy')

        def decode_gated(P4, fpred, pres):
            p = gridpres.decode(gridpres.biased(P4, bias), pres, thr, hi)
            for i in range(len(p)):
                if not fpred[i].any():
                    row = p[i]
                    if (row == 2).any():
                        q = P4[i].reshape(4, 16).copy(); q[2] = -1
                        row[row == 2] = q.argmax(0)[row == 2]
            return p
        g_oof = decode_gated(og, f_oof, po); g_te = decode_gated(tg, f_te, pt)
        print('grid: presence-gated decode', flush=True)
    else:
        g_oof = decode(og, f_oof); g_te = decode(tg, f_te)
    gs = float(np.mean([grid_score(g_oof[i], G[i]) for i in range(ntr)]))
    print('grid oof (constrained)', round(gs, 4), flush=True)

    preds = [dict(semantic_grid=g_oof[i].tolist(), total_price=tot_oof[i],
                  num_items=int(ni_oof[i]),
                  fields_present=dict(zip(FKEYS, [bool(x) for x in f_oof[i]]))) for i in range(ntr)]
    print('OOF BREAKDOWN', {k: round(v, 4) for k, v in breakdown(preds, A).items()}, flush=True)

    rows = [dict(id=te.id[i], answer_json=json.dumps(dict(
        fields_present={k: bool(v) for k, v in zip(FKEYS, f_te[i])},
        num_items=int(ni_te[i]), semantic_grid=[int(x) for x in g_te[i]],
        total_price=tot_te[i]), sort_keys=True)) for i in range(nte)]
    pd.DataFrame(rows).to_csv(W + r'\submission.csv', index=False)
    print('wrote submission.csv', len(rows))


if __name__ == '__main__':
    main(use_beam=bool(int(sys.argv[1])) if len(sys.argv) > 1 else True)
