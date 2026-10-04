"""tokens2 + crop variants in the summary area.

Recall over candidates is the union across crops, so several differently-cropped and
differently-normalised views of the same text give the reader extra chances to get it right.
"""
import os, sys, pickle, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from extract2 import (norm_gray, binarize, blobs_of, est_ch, est_skew, group_lines, split_tokens, SH)
from tokens2 import roi_from_grid, bilin, D, W, GAPS, NF

MAXW = 400
VAR_TIGHT, VAR_LOOSE, VAR_RAW = 1, 2, 3


def crop_var(src, x0, y0, x1, y1, lh, px_m, py_m, lo_p=3, hi_p=97):
    px = max(1.0, lh * px_m); py = max(1.0, lh * py_m)
    H, Wd = src.shape
    a, b = int(max(0, x0 - px)), int(min(Wd, x1 + px))
    c, d = int(max(0, y0 - py)), int(min(H, y1 + py))
    crop = src[c:d, a:b]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 4:
        return None
    lo, hi = np.percentile(crop, lo_p), np.percentile(crop, hi_p)
    f = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    s = SH / f.shape[0]
    w = int(round(f.shape[1] * s))
    if w < 6 or w > MAXW:
        return None
    return (cv2.resize(f, (w, SH),
                       interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA) * 255).astype(np.uint8)


def work(args):
    i, path, gprob = args
    gprob = np.asarray(gprob, np.float32)
    gp = gprob.argmax(0)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, Wd = img.shape
        roi = roi_from_grid(gp, H, Wd)
        norm = norm_gray(path)
        B0 = blobs_of(binarize(norm, 21), H * 0.02, roi)
        ch = est_ch(B0, H)
        ang = est_skew(B0, ch)
        if abs(ang) > 0.4:
            M = cv2.getRotationMatrix2D((Wd / 2, H / 2), ang, 1.0)
            rn = cv2.warpAffine(norm, M, (Wd, H), flags=cv2.INTER_LINEAR, borderValue=255)
            rg = cv2.warpAffine(img, M, (Wd, H), flags=cv2.INTER_LINEAR, borderValue=255)
            pts = np.array([[roi[0], roi[1]], [roi[2], roi[1]],
                            [roi[0], roi[3]], [roi[2], roi[3]]], np.float32)
            p = (M[:, :2] @ pts.T).T + M[:, 2]
            roi2 = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        else:
            rn, rg, roi2, ang = norm, img, roi, 0.0
        B = blobs_of(binarize(rn, ch * 2.2), ch, roi2)
        lines = group_lines(B, ch)
    except Exception:
        return i, [], np.zeros((0, NF + 1), np.float32)
    if not lines:
        return i, [], np.zeros((0, NF + 1), np.float32)
    H, Wd = rn.shape
    rx0, ry0, rx1, ry1 = roi2
    rw = max(rx1 - rx0, 1.0); rh = max(ry1 - ry0, 1.0)
    medlh = float(np.median([L['lh'] for L in lines])); nl = len(lines)
    strips, feats = [], []

    def add(x0, y0, x1, y1, lh, li, kind, ntok, ti, isr, ng):
        yc = (y0 + y1) / 2.0; xc = (x0 + x1) / 2.0
        cp = bilin(gprob, yc, xc, H, Wd)
        base = [(yc - ry0) / rh, (xc - rx0) / rw, (x1 - rx0) / rw, (x1 - x0) / rw,
                lh / max(medlh, 1e-3), ng, kind, li / max(nl - 1, 1),
                (nl - 1 - li) / max(nl - 1, 1), ti, ntok, isr,
                cp[0], cp[1], cp[2], cp[3], yc / H, xc / Wd, lh / H, nl,
                rh / H, rw / Wd, 0.0, (x0 - rx0) / rw, ang, ch / H]
        variants = [(0, rn, 0.18, 0.30, 3, 97)]
        if cp[2] + cp[3] > 0.30:                     # summary area: try harder
            variants += [(VAR_TIGHT, rn, 0.04, 0.10, 2, 98),
                         (VAR_LOOSE, rn, 0.40, 0.60, 5, 95),
                         (VAR_RAW, rg, 0.18, 0.30, 3, 97)]
        for (vid, src, pxm, pym, lo, hi) in variants:
            s = crop_var(src, x0, y0, x1, y1, lh, pxm, pym, lo, hi)
            if s is None:
                continue
            f = list(base); f[22] = s.shape[1] / SH
            feats.append(f + [float(vid)])
            strips.append(s)

    for li, L in enumerate(lines):
        mem = L['boxes']; lh = L['lh']
        y0, y1 = L['y0'], L['y1']
        lx0 = float(mem[:, 0].min()); lx1 = float((mem[:, 0] + mem[:, 2]).max())
        add(lx0, y0, lx1, y1, lh, li, 0, 1, 0, 1.0, len(mem))
        seen = set()
        for gi, gm in enumerate(GAPS):
            groups = split_tokens(mem, lh, gm)
            for ti, gidx in enumerate(groups):
                bx = mem[gidx]
                x0 = float(bx[:, 0].min()); x1 = float((bx[:, 0] + bx[:, 2]).max())
                key = (round(x0), round(x1))
                if key in seen or len(gidx) > 24:
                    continue
                seen.add(key)
                add(x0, float(bx[:, 1].min()), x1, float((bx[:, 1] + bx[:, 3]).max()), lh, li,
                    gi + 1, len(groups), ti, 1.0 if ti == len(groups) - 1 else 0.0, len(gidx))
            if len(groups) >= 2:
                bx = mem[groups[-1]]
                x0 = float(bx[:, 0].min())
                key = (round(x0), round(lx1), 'r')
                if key not in seen:
                    seen.add(key)
                    add(x0, y0, lx1, y1, lh, li, 4, len(groups), len(groups) - 1, 1.0, len(bx))
    return i, strips, (np.array(feats, np.float32) if feats else np.zeros((0, NF + 1), np.float32))


def run(split, gprob_all, df):
    jobs = [(i, os.path.join(D, df.image_path[i].replace('/', os.sep)), gprob_all[i])
            for i in range(len(df))]
    t0 = time.time(); out = {}
    with ProcessPoolExecutor(8) as ex:
        for i, s, f in ex.map(work, jobs, chunksize=4):
            out[i] = (s, f)
    print(split, 'strips', sum(len(v[0]) for v in out.values()),
          'time', round(time.time() - t0, 1), flush=True)
    pickle.dump([out[i] for i in range(len(df))],
                open(W + f'\\cache\\tok3_{split}.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    og = (2 * np.load(W + r'\cache\oof_g.npy') + np.load(W + r'\cache\oof2_g.npy')
          + np.load(W + r'\cache\oof3_g.npy')) / 4
    tg = (2 * np.load(W + r'\cache\te1_g.npy') + np.load(W + r'\cache\te2_g.npy')
          + np.load(W + r'\cache\te3_g.npy')) / 4
    for s in sys.argv[1:] or ['train', 'test']:
        run(s, og if s == 'train' else tg, tr if s == 'train' else te)
