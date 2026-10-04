"""Synthesise the total from the receipt's arithmetic, then keep only what the page supports.

The diagnostic says the reader is close but not exact on 34% of receipts: 11% are one digit
edit from the gold, 21% within two, and only 1.3% are hopeless.  So the answer is on the page
and nearly read -- the lattice just never puts it first, which is why a prefix beam recovered
nothing at all.

A receipt is a ledger, so the total is recoverable from its parts: subtotal + tax, or cash
minus change.  Proposing a+b and b-a over the values actually read yields the true total even
when the total LINE was misread, provided its components were not.  Each proposal must then
also look like something genuinely on the page -- within one or two digit substitutions of a
real reading -- which keeps the pool small and rules out arithmetic coincidences.

Implemented as a set intersection: the substitution neighbourhood is generated from the
(few) read strings, not compared pairwise against the (many) proposals.
"""
import json, pickle, sys, time
from collections import defaultdict
from itertools import combinations
import numpy as np, pandas as pd
from final2 import dg, NUM, C_P2, C_P3

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
DIGITS = '0123456789'


def read_strings(row, thr=0.25, topn=28):
    """Digit strings read anywhere, and the subset from the summary block."""
    texts, confs, F = row
    alls, summ = set(), set()
    m = len(texts)
    if m == 0:
        return alls, summ
    s = F[:m, C_P2] + F[:m, C_P3]
    hot = set(np.where(s >= thr)[0].tolist()) or set(np.argsort(-s)[:topn].tolist())
    for j in range(m):
        for tok in [texts[j].strip()] + NUM.findall(texts[j]):
            d = dg(tok.strip('.,'))
            if 2 <= len(d) <= 9:
                alls.add(d)
                if j in hot:
                    summ.add(d)
    return alls, summ


def neighbourhood(strs, k=2, cap=90):
    """All strings within k digit substitutions of one of `strs`."""
    out = set(strs)
    for s in list(strs)[:cap]:
        L = len(s)
        for i in range(L):
            for c in DIGITS:
                if c != s[i]:
                    out.add(s[:i] + c + s[i + 1:])
        if k >= 2 and L <= 8:
            for i, j in combinations(range(L), 2):
                for c1 in DIGITS:
                    if c1 == s[i]:
                        continue
                    t = s[:i] + c1 + s[i + 1:]
                    for c2 in DIGITS:
                        if c2 != s[j]:
                            out.add(t[:j] + c2 + t[j + 1:])
    return out


def synth_for_image(rows, cap=60):
    allstr, summ = set(), set()
    for r in rows:
        a, s = read_strings(r)
        allstr |= a; summ |= s
    if not summ:
        summ = allstr
    vals = sorted({int(d) for d in summ if len(d) <= 9})
    if len(vals) > 55:
        vals = vals[-55:]
    prop = set()
    for i, a in enumerate(vals):
        for b in vals[i:]:
            for V in (a + b, b - a):
                if 9 < V < 10 ** 9:
                    prop.add(str(V))
    prop -= allstr                                  # already a candidate
    if not prop:
        return {}
    near1 = neighbourhood(summ, k=1)
    near2 = neighbourhood(summ, k=2)
    out = {}
    for s in prop:
        if s in near1:
            out[s] = 1
        elif s in near2:
            out[s] = 2
    if len(out) > cap:
        out = dict(sorted(out.items(), key=lambda kv: kv[1])[:cap])
    return out


def build(split, n, names=('dec4', 'dec5')):
    packs = [pickle.load(open(W + f'\\cache\\{nm}_{split}.pkl', 'rb')) for nm in names]
    res, t0 = [], time.time()
    for i in range(n):
        res.append(synth_for_image([P[i] for P in packs]))
        if i % 100 == 0:
            print(f'  {split} {i}/{n} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(res, open(W + f'\\cache\\synth_{split}.pkl', 'wb'), protocol=4)
    print(f'{split}: {np.mean([len(r) for r in res]):.1f} synthesised values per receipt '
          f'({time.time()-t0:.0f}s)', flush=True)
    return res, packs


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    n = len(golds)
    res, packs = build('train', n)
    base = both = 0
    for i in range(n):
        gd = dg(golds[i])
        pool = set()
        for P in packs:
            for t in P[i][0]:
                for m in [t.strip()] + NUM.findall(t):
                    d = dg(m.strip('.,'))
                    if 2 <= len(d) <= 9:
                        pool.add(d)
        base += gd in pool
        both += (gd in pool) or (gd in res[i])
    print(f'DIGIT RECALL  {base/n:.4f} -> {both/n:.4f}', flush=True)
    te = pd.read_csv(D + r'\test.csv')
    build('test', len(te))
