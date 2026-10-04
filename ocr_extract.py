"""Extract text lines / tokens / glyph patches from receipt images."""
import os
import numpy as np, cv2

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
GP = 20  # glyph patch size


def norm_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    n = cv2.divide(g, bg, scale=255)
    return img, g, n


def binarize(norm, block):
    block = int(block) | 1
    block = max(11, min(75, block))
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 9)
    return th


def comps(th):
    n, lab, st, cen = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return np.zeros((0, 5), np.int32), lab
    return st[1:], lab


def est_char_h(st, H, W):
    if len(st) == 0:
        return H * 0.015
    h = st[:, 3].astype(float); w = st[:, 2].astype(float); a = st[:, 4].astype(float)
    ok = (h > H * 0.005) & (h < H * 0.05) & (w < h * 2.5) & (a > 6)
    if ok.sum() < 10:
        return H * 0.015
    return float(np.median(h[ok]))


def extract(path, roi=None):
    """roi: (x0,y0,x1,y1) in image coords or None -> whole image.
    Returns dict with lines: list of {y0,y1,tokens:[{x0,x1,glyphs:[patch],boxes}]}"""
    img, g, norm = norm_gray(path)
    H, W = g.shape
    st0, _ = comps(binarize(norm, 21))
    if roi is not None and len(st0):
        x0r, y0r, x1r, y1r = roi
        cx = st0[:, 0] + st0[:, 2] / 2.0; cy = st0[:, 1] + st0[:, 3] / 2.0
        st0 = st0[(cx >= x0r) & (cx <= x1r) & (cy >= y0r) & (cy <= y1r)]
    ch = est_char_h(st0, H, W)
    th = binarize(norm, ch * 2.2)
    st, lab = comps(th)
    if len(st) == 0:
        return dict(H=H, W=W, ch=ch, lines=[], img=img, th=th)
    x, y, w, h, a = [st[:, i].astype(float) for i in range(5)]
    fill = a / np.maximum(w * h, 1)
    keep = (h >= max(3, ch * 0.16)) & (h <= ch * 2.6) & (w <= ch * 2.5) & (a >= 4) & (fill > 0.12)
    if roi is not None:
        x0r, y0r, x1r, y1r = roi
        cx = x + w / 2; cy = y + h / 2
        keep &= (cx >= x0r) & (cx <= x1r) & (cy >= y0r) & (cy <= y1r)
    idx = np.where(keep)[0]
    if len(idx) == 0:
        return dict(H=H, W=W, ch=ch, lines=[], img=img, th=th)
    B = st[idx]
    # --- line grouping using only full-height glyphs to define baselines
    big = B[B[:, 3] >= ch * 0.55]
    if len(big) < 2:
        big = B
    yc = big[:, 1] + big[:, 3] / 2.0
    o = np.argsort(yc); big = big[o]; yc = yc[o]
    groups, cur = [], [0]
    for i in range(1, len(big)):
        if yc[i] - yc[cur[-1]] <= max(2.0, ch * 0.55):
            cur.append(i)
        else:
            groups.append(cur); cur = [i]
    groups.append(cur)
    lines = []
    for G in groups:
        cc = big[G]
        y0 = float(cc[:, 1].min()); y1 = float((cc[:, 1] + cc[:, 3]).max())
        lh = y1 - y0
        # attach every kept comp whose centre lies in the band
        cy = B[:, 1] + B[:, 3] / 2.0
        m = (cy >= y0 - lh * 0.25) & (cy <= y1 + lh * 0.25)
        mem = B[m]
        if len(mem) == 0:
            continue
        mem = mem[np.argsort(mem[:, 0])]
        lines.append(dict(y0=y0, y1=y1, lh=lh, boxes=mem))
    # --- tokenise
    out = []
    for L in lines:
        mem = L['boxes']; lh = max(L['lh'], 3.0)
        toks, cur = [], [0]
        for i in range(1, len(mem)):
            prev_end = max(mem[j, 0] + mem[j, 2] for j in cur)
            gap = mem[i, 0] - prev_end
            if gap > lh * 0.55:
                toks.append(cur); cur = [i]
            else:
                cur.append(i)
        toks.append(cur)
        T = []
        for tk in toks:
            bx = mem[tk]
            T.append(dict(x0=float(bx[:, 0].min()), x1=float((bx[:, 0] + bx[:, 2]).max()),
                          y0=float(bx[:, 1].min()), y1=float((bx[:, 1] + bx[:, 3]).max()),
                          boxes=bx))
        out.append(dict(y0=L['y0'], y1=L['y1'], lh=lh, tokens=T))
    return dict(H=H, W=W, ch=ch, lines=out, img=img, th=th, norm=norm)


def glyph_patches(norm, boxes, lh):
    """Grayscale patches, contrast-normalised, GPxGP, plus geometry features."""
    P, F = [], []
    Hn, Wn = norm.shape
    for (x, y, w, h, a) in boxes:
        pad = max(1, int(round(lh * 0.10)))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(Wn, x + w + pad); y1 = min(Hn, y + h + pad)
        crop = norm[y0:y1, x0:x1].astype(np.float32)
        if crop.size == 0:
            crop = np.zeros((3, 3), np.float32)
        lo, hi = np.percentile(crop, 5), np.percentile(crop, 95)
        crop = (crop - lo) / max(hi - lo, 1e-3)
        crop = np.clip(crop, 0, 1)
        # square-pad to keep aspect
        ch_, cw = crop.shape
        s = max(ch_, cw)
        sq = np.ones((s, s), np.float32)
        sq[(s - ch_) // 2:(s - ch_) // 2 + ch_, (s - cw) // 2:(s - cw) // 2 + cw] = crop
        P.append(cv2.resize(sq, (GP, GP), interpolation=cv2.INTER_AREA))
        F.append([h / lh, w / lh, w / max(h, 1), a / max(w * h, 1),
                  ((y + h) - 0) / 1.0, h / max(w, 1)])
    return np.array(P, np.float32), np.array(F, np.float32)
