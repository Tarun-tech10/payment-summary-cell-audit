"""Extract candidate number-token strips + geometric features from every receipt."""
import os, json, sys, pickle, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from ocr_extract import extract, D

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
SH = 32
MAXW = 320


def roi_from_grid(gp, H, Wd, classes=(1, 2, 3), margin=0.10):
    """gp: 4x4 argmax grid -> pixel bbox of the requested classes."""
    m = np.isin(gp, classes)
    if not m.any():
        return (0, 0, Wd, H)
    rs, cs = np.where(m)
    y0 = rs.min() / 4.0; y1 = (rs.max() + 1) / 4.0
    x0 = cs.min() / 4.0; x1 = (cs.max() + 1) / 4.0
    y0 -= margin; y1 += margin; x0 -= margin; x1 += margin
    return (max(0, x0) * Wd, max(0, y0) * H, min(1, x1) * Wd, min(1, y1) * H)


def bilin(prob, yc, xc, H, Wd):
    """prob: 4(cls) x 4 x 4 -> class probs at normalised location."""
    fy = np.clip(yc / H * 4 - 0.5, 0, 3); fx = np.clip(xc / Wd * 4 - 0.5, 0, 3)
    y0, x0 = int(fy), int(fx)
    y1, x1 = min(3, y0 + 1), min(3, x0 + 1)
    ay, ax = fy - y0, fx - x0
    return (prob[:, y0, x0] * (1 - ay) * (1 - ax) + prob[:, y1, x0] * ay * (1 - ax)
            + prob[:, y0, x1] * (1 - ay) * ax + prob[:, y1, x1] * ay * ax)


def strip(norm, tk, lh):
    x0 = int(max(0, tk['x0'] - lh * 0.18)); x1 = int(min(norm.shape[1], tk['x1'] + lh * 0.18))
    y0 = int(max(0, tk['y0'] - lh * 0.28)); y1 = int(min(norm.shape[0], tk['y1'] + lh * 0.28))
    c = norm[y0:y1, x0:x1]
    if c.size == 0 or c.shape[0] < 3:
        return None
    lo, hi = np.percentile(c, 3), np.percentile(c, 97)
    c = np.clip((c.astype(np.float32) - lo) / max(hi - lo, 1e-3), 0, 1)
    s = SH / c.shape[0]
    w = int(round(c.shape[1] * s))
    if w < 6 or w > MAXW:
        return None
    return (cv2.resize(c, (w, SH), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)


def work(args):
    i, path, gprob = args
    gprob = np.asarray(gprob, np.float32)
    gp = gprob.argmax(0)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, Wd = img.shape
        roi = roi_from_grid(gp, H, Wd, (1, 2, 3), 0.12)
        E = extract(path, roi=roi)
    except Exception as e:
        return i, [], np.zeros((0, 24), np.float32)
    norm = E.get('norm')
    if norm is None or not E['lines']:
        return i, [], np.zeros((0, 24), np.float32)
    H, Wd = E['H'], E['W']
    rx0, ry0, rx1, ry1 = roi
    rw = max(rx1 - rx0, 1); rh = max(ry1 - ry0, 1)
    allh = np.array([L['lh'] for L in E['lines']], np.float32)
    medlh = float(np.median(allh)) if len(allh) else 1.0
    nlines = len(E['lines'])
    strips, feats = [], []
    for li, L in enumerate(E['lines']):
        ntok = len(L['tokens'])
        xr = [tk['x1'] for tk in L['tokens']]
        for ti, tk in enumerate(L['tokens']):
            bx = tk['boxes']
            ng = len(bx)
            if ng < 1 or ng > 20:
                continue
            hs = bx[:, 3].astype(np.float32)
            ws_ = bx[:, 2].astype(np.float32)
            # glyph-like: not absurdly wide blobs (barcodes, logos, blur boxes)
            if np.median(ws_) > L['lh'] * 1.4:
                continue
            s = strip(norm, tk, L['lh'])
            if s is None:
                continue
            yc = (tk['y0'] + tk['y1']) / 2.0
            xc = (tk['x0'] + tk['x1']) / 2.0
            cp = bilin(gprob, yc, xc, H, Wd)
            feats.append([
                (yc - ry0) / rh, (xc - rx0) / rw, (tk['x1'] - rx0) / rw,
                (tk['x1'] - tk['x0']) / rw, L['lh'] / max(medlh, 1e-3),
                ng, float(np.median(hs)) / max(L['lh'], 1),
                float(np.std(hs)) / max(L['lh'], 1),
                li / max(nlines - 1, 1), (nlines - 1 - li) / max(nlines - 1, 1),
                ti, ntok, 1.0 if tk['x1'] >= max(xr) - 1 else 0.0,
                cp[0], cp[1], cp[2], cp[3],
                yc / H, xc / Wd, L['lh'] / H, nlines, rh / H, rw / Wd, s.shape[1] / SH,
            ])
            strips.append(s)
    return i, strips, np.array(feats, np.float32)


def run(split, gprob_all, df):
    jobs = [(i, os.path.join(D, df.image_path[i].replace('/', os.sep)), gprob_all[i])
            for i in range(len(df))]
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor(8) as ex:
        for i, s, f in ex.map(work, jobs, chunksize=4):
            out[i] = (s, f)
    print(split, 'tokens', sum(len(v[0]) for v in out.values()),
          'time', round(time.time() - t0, 1), flush=True)
    with open(W + f'\\cache\\tok_{split}.pkl', 'wb') as fh:
        pickle.dump([out[i] for i in range(len(df))], fh, protocol=4)


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    og = np.load(W + r'\cache\oof2_g.npy') if os.path.exists(W + r'\cache\oof2_g.npy') \
        else np.load(W + r'\cache\oof_g.npy')
    tg = np.load(W + r'\cache\te2_g.npy') if os.path.exists(W + r'\cache\te2_g.npy') else None
    run('train', og, tr)
    if tg is not None:
        run('test', tg, te)
