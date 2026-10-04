"""Multi-hypothesis candidate strips (full line + several gap thresholds) per receipt."""
import os, json, sys, pickle, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from extract2 import extract, crop_strip, split_tokens, SH

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
GAPS = (0.45, 0.9, 1.8)
NF = 26


def roi_from_grid(gp, H, Wd, margin=0.12):
    m = np.isin(gp, (1, 2, 3))
    if not m.any():
        return (0.0, 0.0, float(Wd), float(H))
    rs, cs = np.where(m)
    y0 = rs.min() / 4.0 - margin; y1 = (rs.max() + 1) / 4.0 + margin
    x0 = cs.min() / 4.0 - margin; x1 = (cs.max() + 1) / 4.0 + margin
    return (max(0., x0) * Wd, max(0., y0) * H, min(1., x1) * Wd, min(1., y1) * H)


def bilin(prob, yc, xc, H, Wd):
    fy = np.clip(yc / H * 4 - 0.5, 0, 3); fx = np.clip(xc / Wd * 4 - 0.5, 0, 3)
    y0, x0 = int(fy), int(fx); y1, x1 = min(3, y0 + 1), min(3, x0 + 1)
    ay, ax = fy - y0, fx - x0
    return (prob[:, y0, x0] * (1 - ay) * (1 - ax) + prob[:, y1, x0] * ay * (1 - ax)
            + prob[:, y0, x1] * (1 - ay) * ax + prob[:, y1, x1] * ay * ax)


def work(args):
    i, path, gprob = args
    gprob = np.asarray(gprob, np.float32)
    gp = gprob.argmax(0)
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        H, Wd = img.shape
        roi = roi_from_grid(gp, H, Wd)
        rn, lines, ang, ch, roi2 = extract(path, roi)
    except Exception:
        return i, [], np.zeros((0, NF), np.float32)
    if not lines:
        return i, [], np.zeros((0, NF), np.float32)
    H, Wd = rn.shape
    rx0, ry0, rx1, ry1 = roi2
    rw = max(rx1 - rx0, 1.0); rh = max(ry1 - ry0, 1.0)
    medlh = float(np.median([L['lh'] for L in lines]))
    nlines = len(lines)
    strips, feats = [], []

    def add(x0, y0, x1, y1, lh, li, kind, ntok, ti, isright, ng):
        s = crop_strip(rn, x0, y0, x1, y1, lh)
        if s is None:
            return
        yc = (y0 + y1) / 2.0; xc = (x0 + x1) / 2.0
        cp = bilin(gprob, yc, xc, H, Wd)
        feats.append([(yc - ry0) / rh, (xc - rx0) / rw, (x1 - rx0) / rw, (x1 - x0) / rw,
                      lh / max(medlh, 1e-3), ng, kind, li / max(nlines - 1, 1),
                      (nlines - 1 - li) / max(nlines - 1, 1), ti, ntok, isright,
                      cp[0], cp[1], cp[2], cp[3], yc / H, xc / Wd, lh / H, nlines,
                      rh / H, rw / Wd, s.shape[1] / SH, (x0 - rx0) / rw, ang, ch / H])
        strips.append(s)

    for li, L in enumerate(lines):
        mem = L['boxes']; lh = L['lh']
        y0 = L['y0']; y1 = L['y1']
        lx0 = float(mem[:, 0].min()); lx1 = float((mem[:, 0] + mem[:, 2]).max())
        add(lx0, y0, lx1, y1, lh, li, 0, 1, 0, 1.0, len(mem))     # whole line
        seen = set()
        for gi, gm in enumerate(GAPS):
            groups = split_tokens(mem, lh, gm)
            for ti, gidx in enumerate(groups):
                bx = mem[gidx]
                x0 = float(bx[:, 0].min()); x1 = float((bx[:, 0] + bx[:, 2]).max())
                key = (round(x0), round(x1))
                if key in seen:
                    continue
                seen.add(key)
                if len(gidx) > 24:
                    continue
                add(x0, float(bx[:, 1].min()), x1, float((bx[:, 1] + bx[:, 3]).max()),
                    lh, li, gi + 1, len(groups), ti,
                    1.0 if ti == len(groups) - 1 else 0.0, len(gidx))
            # right-hand remainder (label + value merged cases)
            if len(groups) >= 2:
                bx = mem[groups[-1]]
                x0 = float(bx[:, 0].min())
                key = (round(x0), round(lx1), 'r')
                if key not in seen:
                    seen.add(key)
                    add(x0, y0, lx1, y1, lh, li, 4, len(groups), len(groups) - 1, 1.0, len(bx))
    return i, strips, np.array(feats, np.float32) if feats else np.zeros((0, NF), np.float32)


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
                open(W + f'\\cache\\tok2_{split}.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    og = 0.5 * np.load(W + r'\cache\oof_g.npy') + 0.5 * np.load(W + r'\cache\oof2_g.npy')
    tg = 0.5 * np.load(W + r'\cache\te1_g.npy') + 0.5 * np.load(W + r'\cache\te2_g.npy')
    for s in sys.argv[1:] or ['train', 'test']:
        run(s, og if s == 'train' else tg, tr if s == 'train' else te)
