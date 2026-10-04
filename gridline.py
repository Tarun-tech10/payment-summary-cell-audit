"""Reconstruct the 4x4 map by classifying text lines and painting their boxes (max rule)."""
import os, json, pickle, re, sys
import numpy as np, pandas as pd
from scorer import grid_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def cell_masks(boxes, rule='overlap', frac=0.0):
    """boxes: (n,4) normalised x0,y0,x1,y1 -> (n,16) bool coverage of the 4x4 grid."""
    n = len(boxes)
    M = np.zeros((n, 16), bool)
    if n == 0:
        return M
    for r in range(4):
        cy0, cy1 = r / 4, (r + 1) / 4
        for c in range(4):
            cx0, cx1 = c / 4, (c + 1) / 4
            ox = np.minimum(boxes[:, 2], cx1) - np.maximum(boxes[:, 0], cx0)
            oy = np.minimum(boxes[:, 3], cy1) - np.maximum(boxes[:, 1], cy0)
            inter = np.maximum(ox, 0) * np.maximum(oy, 0)
            if rule == 'center':
                bx = (boxes[:, 0] + boxes[:, 2]) / 2; by = (boxes[:, 1] + boxes[:, 3]) / 2
                M[:, r * 4 + c] = (bx >= cx0) & (bx < cx1) & (by >= cy0) & (by < cy1)
            elif frac > 0:
                area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-9)
                M[:, r * 4 + c] = inter / np.minimum(area, 1 / 16.) > frac
            else:
                M[:, r * 4 + c] = inter > 1e-9
    return M


def paint(M, lab):
    if len(lab) == 0:
        return np.zeros(16, np.int64)
    return (M * lab[:, None]).max(0)


def best_labels(M, gold, passes=4):
    n = M.shape[0]
    lab = np.zeros(n, np.int64)
    best = grid_score(paint(M, lab), gold)
    for _ in range(passes):
        ch = False
        for i in range(n):
            cur = lab[i]
            for v in (0, 1, 2, 3):
                if v == cur:
                    continue
                lab[i] = v
                s = grid_score(paint(M, lab), gold)
                if s > best + 1e-9:
                    best, cur, ch = s, v, True
                else:
                    lab[i] = cur
        if not ch:
            break
    return lab, best


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    G = np.array([a['semantic_grid'] for a in A])
    L = pickle.load(open(W + r'\cache\lines_train.pkl', 'rb'))
    for rule, frac in [('overlap', 0.0), ('overlap', 0.05), ('overlap', 0.15), ('center', 0.0)]:
        sc, labs = [], []
        for i in range(len(G)):
            M = cell_masks(L[i][0], rule, frac)
            lab, s = best_labels(M, G[i])
            sc.append(s); labs.append(lab)
        print(f'rule={rule} frac={frac}: oracle grid score {np.mean(sc):.4f}', flush=True)
        if rule == 'overlap' and frac == 0.0:
            pickle.dump(labs, open(W + r'\cache\line_labels.pkl', 'wb'))
