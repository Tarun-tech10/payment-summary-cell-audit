"""Consensus candidates: vote across every reading of the same printed line.

Each physical line is read many times over -- several crop variants, three width scales,
two readers -- and the errors are mostly independent single characters.  A per-position
plurality over those readings recovers strings that no individual reading got right, which
is exactly what the candidate recall ceiling is short of.
"""
import os, json, pickle, sys
from collections import Counter, defaultdict
import numpy as np, pandas as pd
from final2 import dg, reformats, receipt_sep, NUM, C_LI, C_NLINES, C_P2, C_P3

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
PACKS = ['dec6', 'dec7']


def last_num(t):
    m = NUM.findall(t)
    return m[-1].strip('.,') if m else ''


def vote(strings, topk=2):
    """Per-position plurality among equal-length digit strings."""
    out = []
    ds = [dg(s) for s in strings if len(dg(s)) >= 2]
    if not ds:
        return out
    bylen = defaultdict(list)
    for d in ds:
        bylen[len(d)].append(d)
    for L, grp in sorted(bylen.items(), key=lambda kv: -len(kv[1]))[:topk]:
        if len(grp) < 2:
            continue
        cols = [Counter(d[p] for d in grp) for p in range(L)]
        con = ''.join(c.most_common(1)[0][0] for c in cols)
        agree = float(np.mean([c.most_common(1)[0][1] / len(grp) for c in cols]))
        out.append((con, agree, len(grp)))
        # the single most contested position, resolved the other way
        p = int(np.argmin([c.most_common(1)[0][1] / len(grp) for c in cols]))
        alt = cols[p].most_common(2)
        if len(alt) > 1:
            out.append((con[:p] + alt[1][0] + con[p + 1:], agree * 0.5, len(grp)))
    return out


def consensus_for_image(rows):
    """rows = list of (texts, confs, F) for the same image, one per reader pack."""
    groups = defaultdict(list)
    strip_of = {}
    for texts, confs, F in rows:
        if len(texts) == 0:
            continue
        nl = max(F[0, C_NLINES] - 1, 1)
        li = np.round(F[:, C_LI] * nl).astype(int)
        nstrip = len(texts) // 3 if len(texts) % 3 == 0 else len(texts)
        for r, t in enumerate(texts):
            v = last_num(t)
            if v and len(dg(v)) >= 2:
                groups[int(li[r])].append(v)
                strip_of.setdefault(int(li[r]), r % max(nstrip, 1))
    sep = receipt_sep([t for texts, _, _ in rows for t in texts])
    out = []
    for ln, strings in groups.items():
        for con, agree, sup in vote(strings):
            for v in reformats(con, sep):
                out.append((strip_of[ln], v, agree, sup))
    return out


def build_beams(split, n):
    packs = [pickle.load(open(W + f'\\cache\\{nm}_{split}.pkl', 'rb')) for nm in PACKS
             if os.path.exists(W + f'\\cache\\{nm}_{split}.pkl')]
    beams = []
    for i in range(n):
        cons = consensus_for_image([P[i] for P in packs])
        by = defaultdict(list)
        for j, s, agree, sup in cons:
            by[j].append((s, agree))
        beams.append([(j, sorted(v, key=lambda x: -x[1])[:12]) for j, v in by.items()])
    pickle.dump(beams, open(W + f'\\cache\\beamB_{split}.pkl', 'wb'), protocol=4)
    return beams, packs


def recall(packs, golds, beams=None):
    n = len(golds)
    hit_s = hit_d = 0
    for i in range(n):
        pool = set()
        for P in packs:
            texts, _, F = P[i]
            sep = receipt_sep(texts)
            for t in texts:
                for m in [t.strip()] + NUM.findall(t):
                    m = m.strip('.,')
                    d = dg(m)
                    if 2 <= len(d) <= 9:
                        pool.add(m); pool |= reformats(d, sep)
        if beams is not None:
            for j, cl in beams[i]:
                for s, _ in cl:
                    pool.add(s)
        hit_s += golds[i] in pool
        hit_d += dg(golds[i]) in {dg(x) for x in pool}
    return hit_s / n, hit_d / n


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    n = len(golds)
    beams, packs = build_beams('train', n)
    s0, d0 = recall(packs, golds, None)
    s1, d1 = recall(packs, golds, beams)
    print(f'exact-string recall  {s0:.4f} -> {s1:.4f}')
    print(f'digit-string recall  {d0:.4f} -> {d1:.4f}')
    nb = float(np.mean([sum(len(c) for _, c in b) for b in beams]))
    print(f'consensus candidates per image: {nb:.1f}', flush=True)
    te = pd.read_csv(D + r'\test.csv')
    build_beams('test', len(te))
    print('test beams written', flush=True)
