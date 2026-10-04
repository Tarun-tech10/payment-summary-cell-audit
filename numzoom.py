"""Dedicated num_items counter on a zoomed crop of the predicted item area.

In the full-image network an item row is ~10 px tall.  Cropping to the menu region the
grid model already localises and re-rendering it large gives the counter far more signal.
"""
import os, json, time, sys
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from concurrent.futures import ThreadPoolExecutor

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
dev = 'cuda'
NI_MAX = 15
ZH, ZW = 384, 256
LOGF = W + r'\progress_numzoom.txt'


def log(m):
    with open(LOGF, 'a') as f:
        f.write(str(m) + chr(10)); f.flush()
    print(m, flush=True)


def crop_for(path, gp):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, Wd = g.shape
    grid = gp.argmax(0).reshape(4, 4)
    rs, cs = np.where(np.isin(grid, (1, 2, 3)))
    if len(rs) == 0:
        y0, y1, x0, x1 = 0, H, 0, Wd
    else:
        # item area: from the top of the annotated block down to the total band
        r3 = np.where((grid == 3).any(1))[0]
        top = rs.min() / 4.0
        bot = ((r3.max() + 1) / 4.0) if len(r3) else (rs.max() + 1) / 4.0
        top = max(0.0, top - 0.10); bot = min(1.0, bot + 0.06)
        x0 = max(0.0, cs.min() / 4.0 - 0.06) * Wd
        x1 = min(1.0, (cs.max() + 1) / 4.0 + 0.06) * Wd
        y0, y1 = top * H, bot * H
    y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
    if y1 - y0 < 16 or x1 - x0 < 16:
        y0, y1, x0, x1 = 0, H, 0, Wd
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    norm = cv2.divide(g, bg, scale=255)
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = np.stack([
        cv2.resize(norm[y0:y1, x0:x1], (ZW, ZH), interpolation=cv2.INTER_AREA),
        cv2.resize(th[y0:y1, x0:x1], (ZW, ZH), interpolation=cv2.INTER_AREA),
        cv2.resize(g[y0:y1, x0:x1], (ZW, ZH), interpolation=cv2.INTER_AREA)], -1)
    return out


def build(df, gp):
    paths = [os.path.join(D, p.replace('/', os.sep)) for p in df.image_path]
    with ThreadPoolExecutor(8) as ex:
        arrs = list(ex.map(lambda a: crop_for(*a), zip(paths, gp)))
    return np.stack(arrs).astype(np.uint8)


class Counter(nn.Module):
    def __init__(s, cin=3, w=(32, 64, 96, 160, 224)):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(True), nn.Conv2d(o, o, 3, 1, 1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        s.f = nn.Sequential(nn.Conv2d(cin, w[0], 5, 2, 2, bias=False), nn.BatchNorm2d(w[0]),
                            nn.ReLU(True), blk(w[0], w[1]), blk(w[1], w[2]),
                            blk(w[2], w[3]), blk(w[3], w[4]))
        s.drop = nn.Dropout(0.3)
        s.fc = nn.Linear(w[4] * 2, NI_MAX + 1)

    def forward(s, x):
        h = s.f(x)
        # rows carry the count signal: pool columns, then summarise rows two ways
        v = h.mean(3)
        z = torch.cat([v.mean(2), v.max(2).values], 1)
        return s.fc(s.drop(z))


def aug(x, rng):
    B = x.shape[0]
    a = torch.rand(B, 1, 1, 1, device=x.device) * 0.5 + 0.75
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.25
    y = (x * a + b).clamp(0, 1)
    y = y + torch.randn_like(y) * 0.02
    ang = (torch.rand(B, device=x.device) - .5) * (2 * np.pi / 180 * 2.5)
    sy = 1 + (torch.rand(B, device=x.device) - .5) * 0.10
    sx = 1 + (torch.rand(B, device=x.device) - .5) * 0.10
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = torch.cos(ang) / sx; th[:, 0, 1] = -torch.sin(ang)
    th[:, 1, 0] = torch.sin(ang); th[:, 1, 1] = torch.cos(ang) / sy
    th[:, 0, 2] = (torch.rand(B, device=x.device) - .5) * 0.06
    th[:, 1, 2] = (torch.rand(B, device=x.device) - .5) * 0.06
    g = F.affine_grid(th, y.shape, align_corners=False)
    return F.grid_sample(y, g, padding_mode='border', align_corners=False)


def main(epochs=90, bs=16, lr=2.5e-3):
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    y = np.array([min(a['num_items'], NI_MAX) for a in A], np.int64)
    og = np.load(W + r'\cache\oofC_g.npy'); tg = np.load(W + r'\cache\teC_g.npy')
    ctr = W + r'\cache\zoom_train.npy'; cte = W + r'\cache\zoom_test.npy'
    if not os.path.exists(ctr):
        np.save(ctr, build(tr, og)); np.save(cte, build(te, tg))
    X = torch.from_numpy(np.load(ctr)).float().permute(0, 3, 1, 2) / 255.
    Xe = torch.from_numpy(np.load(cte)).float().permute(0, 3, 1, 2) / 255.
    log(f'zoom cache {tuple(X.shape)}')
    Y = torch.from_numpy(y)
    oof = np.zeros((len(X), NI_MAX + 1), np.float32)
    pte = np.zeros((len(Xe), NI_MAX + 1), np.float32)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for k, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0)
                                 .split(np.zeros(len(X)), np.minimum(y, 8))):
        torch.manual_seed(k)
        net = Counter().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
        n = len(ti); steps = max(1, n // bs)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.25)
        scaler = torch.amp.GradScaler('cuda')
        Xt = X[ti].to(dev); Yt = Y[ti].to(dev)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n, device=dev)
            for i in range(steps):
                idx = perm[i * bs:(i + 1) * bs]
                xb = aug(Xt[idx], rng)
                with torch.amp.autocast('cuda'):
                    loss = F.cross_entropy(net(xb), Yt[idx], label_smoothing=0.05)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        net.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for src, dst, ix in ((X[vi], oof, vi), (Xe, pte, None)):
                acc = []
                for i in range(0, len(src), 16):
                    acc.append(net(src[i:i + 16].to(dev)).float().softmax(1).cpu().numpy())
                p = np.concatenate(acc)
                if ix is not None:
                    dst[ix] = p
                else:
                    dst += p / 5
        del Xt, Yt, net
        torch.cuda.empty_cache()
        log(f'fold {k} acc so far {(oof[vi].argmax(1) == y[vi]).mean():.4f} ({time.time()-t0:.0f}s)')
    np.save(W + r'\cache\ni_zoom_oof.npy', oof)
    np.save(W + r'\cache\ni_zoom_te.npy', pte)
    base = np.load(W + r'\cache\oof2_n.npy')
    log(f'zoom acc {(oof.argmax(1) == y).mean():.4f}  full-image acc {(base.argmax(1) == y).mean():.4f}')
    for w in [0, .25, .5, .75, 1]:
        log(f'  blend {w}: {((w * oof + (1 - w) * base).argmax(1) == y).mean():.4f}')


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 90)
