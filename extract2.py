"""Deskewed, multi-hypothesis strip extraction for receipt OCR."""
import os
import numpy as np, cv2

SH = 32
MAXW = 400


def norm_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    return cv2.divide(img, bg, scale=255)


def binarize(norm, block):
    b = int(block) | 1
    b = max(11, min(75, b))
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, b, 9)
    return cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def blobs_of(th, ch, roi=None):
    n, lab, st, _ = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return np.zeros((0, 5), np.int32)
    B = st[1:]
    x, y, w, h, a = [B[:, i].astype(float) for i in range(5)]
    fill = a / np.maximum(w * h, 1)
    keep = (h >= max(3, ch * 0.16)) & (h <= ch * 2.8) & (w <= ch * 3.0) & (a >= 4) & (fill > 0.10)
    if roi is not None:
        x0, y0, x1, y1 = roi
        cx = x + w / 2; cy = y + h / 2
        keep &= (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
    return B[keep]


def est_ch(B, H):
    if len(B) == 0:
        return H * 0.015
    h = B[:, 3].astype(float)
    ok = (h > H * 0.005) & (h < H * 0.05)
    return float(np.median(h[ok])) if ok.sum() >= 8 else H * 0.015


def est_skew(B, ch):
    """Angle (deg) that maximises the sharpness of the horizontal projection."""
    if len(B) < 12:
        return 0.0
    cx = B[:, 0] + B[:, 2] / 2.0
    cy = B[:, 1] + B[:, 3] / 2.0
    cx = cx - cx.mean(); cy = cy - cy.mean()
    best, ba = -1, 0.0
    binw = max(2.0, ch * 0.45)
    for a in np.arange(-8, 8.01, 0.5):
        r = np.deg2rad(a)
        yp = cy * np.cos(r) - cx * np.sin(r)
        hist, _ = np.histogram(yp, bins=int((yp.max() - yp.min()) / binw) + 2)
        v = float((hist.astype(float) ** 2).sum())
        if v > best:
            best, ba = v, float(a)
    return ba


def rotate(img, ang):
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255), M


def group_lines(B, ch):
    if len(B) == 0:
        return []
    big = B[B[:, 3] >= ch * 0.55]
    if len(big) < 2:
        big = B
    yc = big[:, 1] + big[:, 3] / 2.0
    o = np.argsort(yc); big = big[o]; yc = yc[o]
    out, cur = [], [0]
    for i in range(1, len(big)):
        if yc[i] - yc[cur[-1]] <= max(2.0, ch * 0.6):
            cur.append(i)
        else:
            out.append(cur); cur = [i]
    out.append(cur)
    lines = []
    ally = B[:, 1] + B[:, 3] / 2.0
    for G in out:
        cc = big[G]
        y0 = float(cc[:, 1].min()); y1 = float((cc[:, 1] + cc[:, 3]).max())
        lh = max(y1 - y0, 3.0)
        m = (ally >= y0 - lh * 0.30) & (ally <= y1 + lh * 0.30)
        mem = B[m]
        if len(mem) == 0:
            continue
        mem = mem[np.argsort(mem[:, 0])]
        lines.append(dict(y0=y0, y1=y1, lh=lh, boxes=mem))
    return lines


def split_tokens(mem, lh, gap_mul):
    groups, cur = [], [0]
    end = mem[0, 0] + mem[0, 2]
    for i in range(1, len(mem)):
        if mem[i, 0] - end > lh * gap_mul:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
        end = max(end, mem[i, 0] + mem[i, 2])
    groups.append(cur)
    return groups


def crop_strip(norm, x0, y0, x1, y1, lh):
    px = max(1.0, lh * 0.18); py = max(1.0, lh * 0.30)
    H, W = norm.shape
    a = int(max(0, x0 - px)); b = int(min(W, x1 + px))
    c = int(max(0, y0 - py)); d = int(min(H, y1 + py))
    crop = norm[c:d, a:b]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 4:
        return None
    lo, hi = np.percentile(crop, 3), np.percentile(crop, 97)
    f = np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    s = SH / f.shape[0]
    w = int(round(f.shape[1] * s))
    if w < 6 or w > MAXW:
        return None
    return (cv2.resize(f, (w, SH), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
            * 255).astype(np.uint8)


def extract(path, roi=None):
    """Returns (norm_rotated, lines, angle). Lines hold boxes in rotated coords."""
    norm = norm_gray(path)
    H, W = norm.shape
    th0 = binarize(norm, 21)
    B0 = blobs_of(th0, H * 0.02, roi)
    ch = est_ch(B0, H)
    ang = est_skew(B0, ch)
    if abs(ang) > 0.4:
        rn, M = rotate(norm, ang)
        if roi is not None:
            pts = np.array([[roi[0], roi[1]], [roi[2], roi[1]],
                            [roi[0], roi[3]], [roi[2], roi[3]]], np.float32)
            p = (M[:, :2] @ pts.T).T + M[:, 2]
            roi2 = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
        else:
            roi2 = None
    else:
        rn, roi2, ang = norm, roi, 0.0
    th = binarize(rn, ch * 2.2)
    B = blobs_of(th, ch, roi2)
    lines = group_lines(B, ch)
    return rn, lines, ang, ch, (roi2 or (0, 0, W, H))
