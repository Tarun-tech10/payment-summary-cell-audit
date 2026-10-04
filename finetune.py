"""Domain-adapt the CRNN on real receipt strips (gold-verified + confident pseudo-labels)."""
import os, pickle, json, re, time, random, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn
from crnn import CRNN, pad_batch, dev
from crnn2 import encode2, decode2
from synth2 import NCLS2, C2I2

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
OK = re.compile(r'^[-]?[0-9A-Za-z][0-9A-Za-z.,:%]*$')


def build_real():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    X, Y, Wt = [], [], []
    for split in ['train', 'test']:
        tok = pickle.load(open(W + f'\\cache\\tok2_{split}.pkl', 'rb'))
        dec = pickle.load(open(W + f'\\cache\\dec2_{split}.pkl', 'rb'))
        for i, ((strips, F), (texts, confs, _)) in enumerate(zip(tok, dec)):
            g = golds[i] if split == 'train' else None
            for s, t, c in zip(strips, texts, confs):
                t = t.strip()
                if not t or len(t) > 22 or not OK.match(t):
                    continue
                if not all(ch in C2I2 for ch in t):
                    continue
                if g is not None and t == g:
                    X.append(s); Y.append(t); Wt.append(4)
                elif c > -0.06 and len(t) >= 2:
                    X.append(s); Y.append(t); Wt.append(1)
    return X, Y, Wt


def jitter(im, rng):
    a = im.astype(np.float32) / 255.
    if rng.random() < 0.5:
        a = np.clip(a * rng.uniform(0.8, 1.2) + rng.uniform(-0.12, 0.12), 0, 1)
    if rng.random() < 0.3:
        import cv2
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        a = np.clip(a + rng.normal(0, rng.uniform(0.01, 0.05), a.shape), 0, 1)
    return (a * 255).astype(np.uint8)


def main(epochs=3, bs=96, lr=4e-4):
    Xr, Yr, Wr = build_real()
    print('real strips', len(Xr), 'gold-verified', sum(1 for w in Wr if w == 4), flush=True)
    with open(W + r'\cache\synth2.pkl', 'rb') as f:
        Xs, Ys = pickle.load(f)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(Xs), size=min(90000, len(Xs)), replace=False)
    X = list(Xr) + [Xs[i] for i in sel]
    Y = list(Yr) + [Ys[i] for i in sel]
    Wt = list(Wr) + [1] * len(sel)
    rep = [i for i, w in enumerate(Wt) for _ in range(w)]
    print('total training items (with repeats)', len(rep), flush=True)
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + r'\cache\crnn2.pt', map_location=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    steps = len(rep) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.2)
    jr = np.random.default_rng(1)
    t0 = time.time()
    for ep in range(epochs):
        order = sorted(rep, key=lambda i: X[i].shape[1])
        buckets = [order[k * bs:(k + 1) * bs] for k in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0.0
        for idx in buckets:
            xb, ws = pad_batch([jitter(X[i], jr) for i in idx])
            xb = torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)
            tgt, tl = encode2([Y[i] for i in idx])
            with torch.amp.autocast('cuda'):
                lg = net(xb)
            lp = lg.float().log_softmax(-1).permute(1, 0, 2)
            il = torch.full((len(idx),), lp.shape[0], dtype=torch.long)
            loss = ctc(lp, tgt, il, tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        print(f'ft ep {ep} loss {tot/max(steps,1):.4f} {time.time()-t0:.0f}s', flush=True)
        torch.save(net.state_dict(), W + r'\cache\crnn_ft.pt')


if __name__ == '__main__':
    main(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
