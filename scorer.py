import json, re
import numpy as np

def _digits(s):
    return re.sub(r'\D', '', s)

def total_price_score(pred, gold):
    if pred.strip() == gold.strip():
        return 1.0
    p = _digits(pred); g = _digits(gold)
    if len(p) == 0 and len(g) == 0:
        return 0.30
    # char f1 over digits (multiset)
    from collections import Counter
    cp, cg = Counter(p), Counter(g)
    inter = sum((cp & cg).values())
    if len(p) == 0 or len(g) == 0:
        f1 = 0.0
    else:
        prec = inter/len(p); rec = inter/len(g)
        f1 = 0.0 if prec+rec == 0 else 2*prec*rec/(prec+rec)
    L = max(len(p), len(g))
    if L == 0:
        pos = 0.0
    else:
        pos = sum(1 for i in range(min(len(p), len(g))) if p[i] == g[i]) / L
    return 0.30*(0.65*f1 + 0.35*pos)

def grid_score(pred, gold):
    p = np.asarray(pred); g = np.asarray(gold)
    vals = sorted(set(p[p > 0].tolist()) | set(g[g > 0].tolist()))
    if not vals:
        macro_iou = 1.0
    else:
        ious = []
        for v in vals:
            pi = (p == v); gi = (g == v)
            u = (pi | gi).sum()
            ious.append(((pi & gi).sum()/u) if u else 1.0)
        macro_iou = float(np.mean(ious))
    pa = (p > 0); ga = (g > 0)
    inter = (pa & ga).sum()
    if pa.sum() + ga.sum() == 0:
        af1 = 1.0
    else:
        af1 = 2*inter/(pa.sum()+ga.sum())
    return 0.70*macro_iou + 0.30*af1

def fields_score(pred, gold):
    keys = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
    m = [bool(pred[k]) == bool(gold[k]) for k in keys]
    if all(m):
        return 1.0
    return 0.25*(sum(m)/4.0)

def row_score(pred, gold):
    return (0.45*grid_score(pred['semantic_grid'], gold['semantic_grid'])
            + 0.35*total_price_score(pred['total_price'], gold['total_price'])
            + 0.12*(1.0 if int(pred['num_items']) == int(gold['num_items']) else 0.0)
            + 0.08*fields_score(pred['fields_present'], gold['fields_present']))

def breakdown(preds, golds):
    g = np.mean([grid_score(p['semantic_grid'], q['semantic_grid']) for p, q in zip(preds, golds)])
    t = np.mean([total_price_score(p['total_price'], q['total_price']) for p, q in zip(preds, golds)])
    n = np.mean([1.0 if int(p['num_items']) == int(q['num_items']) else 0.0 for p, q in zip(preds, golds)])
    f = np.mean([fields_score(p['fields_present'], q['fields_present']) for p, q in zip(preds, golds)])
    tot = 0.45*g + 0.35*t + 0.12*n + 0.08*f
    return dict(grid=g, total=t, num_items=n, fields=f, score=tot)
