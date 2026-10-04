"""Text-component extraction + receipt region detection for receipt images."""
import os
import numpy as np
import cv2

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(path)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, g


def text_mask(gray):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    norm = cv2.divide(gray, bg, scale=255)
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return th, norm


def char_blobs(th, H, W):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    if n <= 1:
        return np.zeros((0, 5), np.float32)
    s = stats[1:].astype(np.float32)
    x, y, w, h, a = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    minh = max(4.0, H * 0.004)
    maxh = H * 0.035
    ok = (h >= minh) & (h <= maxh) & (w <= H * 0.06) & (w >= 1) & (a >= 5)
    fill = a / np.maximum(w * h, 1)
    ok &= (fill > 0.15) & (fill < 0.98)
    ok &= (w / np.maximum(h, 1) < 3.0)
    return np.stack([x, y, w, h, a], 1)[ok]


def receipt_region(blobs, H, W):
    """Hysteresis density-map localisation of the receipt's text block."""
    if len(blobs) < 8:
        return (0, 0, W, H), None
    S = 8
    hh, ww = H // S + 1, W // S + 1
    dens = np.zeros((hh, ww), np.float32)
    cx = ((blobs[:, 0] + blobs[:, 2] / 2) / S).astype(int).clip(0, ww - 1)
    cy = ((blobs[:, 1] + blobs[:, 3] / 2) / S).astype(int).clip(0, hh - 1)
    np.add.at(dens, (cy, cx), 1.0)
    kx = int(round(W * 0.05 / S)) * 2 + 1
    ky = int(round(H * 0.025 / S)) * 2 + 1
    sm = cv2.GaussianBlur(dens, (kx, ky), 0)
    mx = float(sm.max())
    if mx <= 0:
        return (0, 0, W, H), None
    low = (sm > mx * 0.06).astype(np.uint8)
    high = sm > mx * 0.35
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(low, 8)
    if n <= 1:
        return (0, 0, W, H), None
    keep = set(np.unique(lab[high])) - {0}
    if not keep:
        keep = {1 + int(np.argmax([sm[lab == i].sum() for i in range(1, n)]))}
    m = np.isin(lab, list(keep))
    ys, xs = np.where(m)
    box = (max(0, int(xs.min()) * S - 6), max(0, int(ys.min()) * S - 6),
           min(W, int(xs.max() + 1) * S + 6), min(H, int(ys.max() + 1) * S + 6))
    return box, m


def in_box(blobs, box):
    x0, y0, x1, y1 = box
    cx = blobs[:, 0] + blobs[:, 2] / 2
    cy = blobs[:, 1] + blobs[:, 3] / 2
    return blobs[(cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)]


def group_lines(blobs):
    """Group char blobs into text lines by y proximity."""
    if len(blobs) == 0:
        return []
    hmed = float(np.median(blobs[:, 3]))
    yc = blobs[:, 1] + blobs[:, 3] / 2
    order = np.argsort(yc)
    b = blobs[order]; yc = yc[order]
    lines, cur = [], [0]
    for i in range(1, len(b)):
        if yc[i] - yc[cur[-1]] <= max(2.0, hmed * 0.55):
            cur.append(i)
        else:
            lines.append(cur); cur = [i]
    lines.append(cur)
    out = []
    for L in lines:
        cc = b[L]
        out.append(dict(x0=float(cc[:, 0].min()), x1=float((cc[:, 0] + cc[:, 2]).max()),
                        y0=float(cc[:, 1].min()), y1=float((cc[:, 1] + cc[:, 3]).max()),
                        yc=float(np.median(cc[:, 1] + cc[:, 3] / 2)),
                        n=len(L), h=float(np.median(cc[:, 3])),
                        ink=float(cc[:, 4].sum()), blobs=cc))
    return out


def process(path):
    img, g = load_gray(path)
    H, W = g.shape
    th, norm = text_mask(g)
    blobs = char_blobs(th, H, W)
    box, _ = receipt_region(blobs, H, W)
    rb = in_box(blobs, box)
    lines = group_lines(rb)
    return dict(img=img, gray=g, norm=norm, th=th, blobs=blobs, box=box,
                rblobs=rb, lines=lines, H=H, W=W)
