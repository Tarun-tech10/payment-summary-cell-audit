"""Per-line boxes in ORIGINAL image coordinates, in the same order tokens2.py emits lines."""
import os, sys, pickle, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor
from extract2 import (norm_gray, binarize, blobs_of, est_ch, est_skew, group_lines)
from tokens2 import roi_from_grid, D, W


def work(args):
    i, path, gprob = args
    gp = np.asarray(gprob, np.float32).argmax(0)
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
            pts = np.array([[roi[0], roi[1]], [roi[2], roi[1]],
                            [roi[0], roi[3]], [roi[2], roi[3]]], np.float32)
            p = (M[:, :2] @ pts.T).T + M[:, 2]
            roi2 = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
            Minv = cv2.invertAffineTransform(M)
        else:
            rn, roi2, Minv = norm, roi, None
        B = blobs_of(binarize(rn, ch * 2.2), ch, roi2)
        lines = group_lines(B, ch)
    except Exception:
        return i, np.zeros((0, 4), np.float32), (0., 0., 1., 1.)
    out = []
    for L in lines:
        mem = L['boxes']
        x0 = float(mem[:, 0].min()); x1 = float((mem[:, 0] + mem[:, 2]).max())
        y0 = L['y0']; y1 = L['y1']
        if Minv is not None:
            c = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], np.float32)
            p = (Minv[:, :2] @ c.T).T + Minv[:, 2]
            x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
            y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        out.append([x0 / Wd, y0 / H, x1 / Wd, y1 / H])
    # roi back in original normalised coords
    r = (roi[0] / Wd, roi[1] / H, roi[2] / Wd, roi[3] / H)
    return i, np.array(out, np.float32) if out else np.zeros((0, 4), np.float32), r


def run(split, gprob, df):
    jobs = [(i, os.path.join(D, df.image_path[i].replace('/', os.sep)), gprob[i])
            for i in range(len(df))]
    t0 = time.time(); res = {}
    with ProcessPoolExecutor(8) as ex:
        for i, b, r in ex.map(work, jobs, chunksize=4):
            res[i] = (b, r)
    print(split, 'lines', sum(len(res[i][0]) for i in res), round(time.time() - t0, 1), 's')
    pickle.dump([res[i] for i in range(len(df))],
                open(W + f'\\cache\\lines_{split}.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    og = (2 * np.load(W + r'\cache\oof_g.npy') + np.load(W + r'\cache\oof2_g.npy')
          + np.load(W + r'\cache\oof3_g.npy')) / 4
    tg = (2 * np.load(W + r'\cache\te1_g.npy') + np.load(W + r'\cache\te2_g.npy')
          + np.load(W + r'\cache\te3_g.npy')) / 4
    for s in sys.argv[1:] or ['train', 'test']:
        run(s, og if s == 'train' else tg, tr if s == 'train' else te)
