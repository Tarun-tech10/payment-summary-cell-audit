"""Close the synthetic->real gap by consistency training on real receipt crops.

No gold labels are used, so this stays fold-honest and can also see the test crops.
A crop whose reading is identical at all three width scales is treated as a pseudo-label;
the network is then asked to reproduce that reading from a heavily degraded version of the
same crop.  The supervision is the model's own confident output, so nothing is memorised --
what transfers is robustness on real thermal print rather than on rendered glyphs.
"""
import os, sys, pickle, time, math, random
from collections import Counter
import numpy as np, cv2, torch
import torch.nn as nn
from crnn import CRNN, pad_batch, dev, SH
from crnn2 import encode2
from synth2 import NCLS2, C2I2

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
LOGF = W + r'\progress_adapt.txt'


def log(m):
    with open(LOGF, 'a') as f:
        f.write(str(m) + chr(10)); f.flush()
    print(m, flush=True)


def harvest(split, dec='dec6', min_len=2, max_len=16):
    """Crops whose reading is stable across all three width scales."""
    tok = pickle.load(open(W + f'\\cache\\tok3_{split}.pkl', 'rb'))
    pack = pickle.load(open(W + f'\\cache\\{dec}_{split}.pkl', 'rb'))
    X, Y, C = [], [], []
    for (strips, _), (texts, confs, _) in zip(tok, pack):
        n = len(strips)
        if n == 0 or len(texts) != 3 * n:
            continue
        for j in range(n):
            a, b, c = texts[j], texts[j + n], texts[j + 2 * n]
            if a != b or b != c:
                continue
            t = a.strip()
            if not (min_len <= len(t) <= max_len) or not any(ch.isdigit() for ch in t):
                continue
            if any(ch not in C2I2 for ch in t):
                continue
            X.append(strips[j]); Y.append(t)
            C.append(float(np.mean([confs[j], confs[j + n], confs[j + 2 * n]])))
    return X, Y, np.array(C, np.float32)


def degrade_real(im, rng):
    """Push the crop further than the reader currently copes with."""
    h, w = im.shape
    f = rng.uniform(0.55, 1.25)
    nw = max(8, int(round(w * f)))
    im = cv2.resize(im, (nw, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        sh = rng.uniform(0.45, 1.0)
        small = cv2.resize(im, (max(6, int(nw * sh)), max(8, int(h * sh))),
                           interpolation=cv2.INTER_AREA)
        im = cv2.resize(small, (nw, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.45:
        k = rng.choice([3, 5])
        im = cv2.GaussianBlur(im, (k, k), 0)
    if rng.random() < 0.35:
        kx = np.ones((1, rng.choice([2, 3])), np.uint8)
        im = (cv2.erode if rng.random() < 0.5 else cv2.dilate)(im, kx)
    a = rng.uniform(0.55, 1.35); b = rng.uniform(-35, 35)
    im = np.clip(im.astype(np.float32) * a + b, 0, 255)
    if rng.random() < 0.5:
        im += rng.normal(0, rng.uniform(3, 16), im.shape)
    if rng.random() < 0.3:                       # thermal exposure banding
        band = np.sin(np.linspace(0, rng.uniform(2, 9), im.shape[0]))[:, None]
        im += band * rng.uniform(8, 30)
    return np.clip(im, 0, 255).astype(np.uint8)


def main(epochs=3, bs=64, lr=3e-4, base='crnn3.pt', out='crnn5.pt', synth_frac=0.35):
    rng = np.random.default_rng(0)
    t0 = time.time()
    Xr, Yr, Cr = [], [], []
    for sp in ('train', 'test'):
        a, b, c = harvest(sp)
        Xr += a; Yr += b; Cr.append(c)
        log(f'{sp}: {len(a)} scale-stable crops')
    Cr = np.concatenate(Cr)
    thr = float(np.quantile(Cr, 0.25))
    keep = np.where(Cr >= thr)[0]
    Xr = [Xr[i] for i in keep]; Yr = [Yr[i] for i in keep]
    log(f'kept {len(Xr)} pseudo-labelled real crops (conf >= {thr:.3f}); '
        f'{time.time()-t0:.0f}s')
    log('sample labels: ' + repr([Yr[i] for i in rng.integers(0, len(Yr), 12)]))

    with open(W + r'\cache\synth3.pkl', 'rb') as f:
        Xs, Ys = pickle.load(f)
    log(f'synth pool {len(Xs)}')

    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + base, map_location=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    nreal = len(Xr)
    steps = nreal // int(bs * (1 - synth_frac))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.2)
    nsyn = int(bs * synth_frac); nrl = bs - nsyn
    for ep in range(epochs):
        order = np.argsort([x.shape[1] for x in Xr])     # width buckets keep padding small
        buckets = [order[i * nrl:(i + 1) * nrl] for i in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0.0
        for bi, idx in enumerate(buckets):
            imgs = [degrade_real(Xr[i], rng) for i in idx]
            txts = [Yr[i] for i in idx]
            si = rng.integers(0, len(Xs), nsyn)
            imgs += [Xs[i] for i in si]; txts += [Ys[i] for i in si]
            xb, ws = pad_batch(imgs)
            xb = torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)
            tgt, tl = encode2(txts)
            with torch.amp.autocast('cuda'):
                lg = net(xb)
            lp = lg.float().log_softmax(-1).permute(1, 0, 2)
            il = torch.full((len(txts),), lp.shape[0], dtype=torch.long)
            loss = ctc(lp, tgt, il, tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
            if bi % 200 == 0:
                log(f'  ep{ep} step {bi}/{steps} loss {tot/max(bi,1):.4f} '
                    f'({time.time()-t0:.0f}s)')
        log(f'ep {ep} loss {tot/steps:.4f} ({time.time()-t0:.0f}s)')
    torch.save(net.state_dict(), W + '\\cache\\' + out)
    log(f'saved {out}')


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
