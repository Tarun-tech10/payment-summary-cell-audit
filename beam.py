"""CTC prefix beam search restricted to money symbols, rescored with a format prior.

Most reader mistakes on the true total are single-character slips ("46o0%" for "40,000").
Searching only over digits/separators and preferring the digit-grouping patterns that
actually occur on receipts converts a lot of those near-misses into exact hits.
"""
import re, json, math
from collections import defaultdict
import numpy as np

MONEY = '0123456789,.'


def build_prior(train_totals):
    """P(pattern) over digit/separator shapes, e.g. 'DD,DDD'."""
    from collections import Counter
    c = Counter()
    for t in train_totals:
        p = ''.join('D' if ch.isdigit() else ch for ch in t.strip() if ch in MONEY)
        if p:
            c[p] += 1
    n = sum(c.values())
    return {k: math.log((v + 0.5) / (n + 50.0)) for k, v in c.items()}, math.log(0.5 / (n + 50.0))


def prefix_beam(logp, allowed_idx, idx2char, beam=12, topk=6, blank=0):
    """logp: (T, C) log-softmax.  Returns [(string, logprob)] best-first."""
    T = logp.shape[0]
    cols = [blank] + list(allowed_idx)
    lp = logp[:, cols]
    # renormalise over the restricted symbol set
    m = lp.max(1, keepdims=True)
    lp = lp - (m + np.log(np.exp(lp - m).sum(1, keepdims=True)))
    chars = [''] + [idx2char[i] for i in allowed_idx]
    NEG = -1e30
    # beams: prefix -> (p_blank, p_nonblank)
    beams = {'': (0.0, NEG)}
    for t in range(T):
        row = lp[t]
        nxt = defaultdict(lambda: [NEG, NEG])
        # keep only symbols with meaningful mass
        cand = [k for k in range(len(cols)) if row[k] > -9.0]
        if not cand:
            cand = [int(np.argmax(row))]
        for pre, (pb, pnb) in beams.items():
            ptot = np.logaddexp(pb, pnb)
            for k in cand:
                p = row[k]
                if k == 0:
                    e = nxt[pre]
                    e[0] = np.logaddexp(e[0], ptot + p)
                    continue
                c = chars[k]
                if pre and pre[-1] == c:
                    e = nxt[pre]
                    e[1] = np.logaddexp(e[1], pnb + p)      # repeat, no new char
                    np_ = pre + c
                    e2 = nxt[np_]
                    e2[1] = np.logaddexp(e2[1], pb + p)     # separated by a blank
                else:
                    np_ = pre + c
                    e2 = nxt[np_]
                    e2[1] = np.logaddexp(e2[1], ptot + p)
        scored = sorted(nxt.items(), key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))
        beams = {k: tuple(v) for k, v in scored[:beam]}
    out = [(k, float(np.logaddexp(v[0], v[1]))) for k, v in beams.items() if k]
    out.sort(key=lambda kv: -kv[1])
    return out[:topk]


def rescore(cands, prior, default, lam=0.55):
    out = []
    for s, lp in cands:
        p = ''.join('D' if ch.isdigit() else ch for ch in s)
        out.append((s, lp + lam * prior.get(p, default)))
    out.sort(key=lambda kv: -kv[1])
    return out
