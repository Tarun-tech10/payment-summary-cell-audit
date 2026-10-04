"""CRNN over the extended (word+number) alphabet."""
import os, pickle, time, math, random, sys
import numpy as np, torch
import torch.nn as nn
from crnn import CRNN, pad_batch, dev, SH
from synth2 import ALPHA2, C2I2, I2C2, NCLS2

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def encode2(txts):
    tgt, lens = [], []
    for t in txts:
        e = [C2I2[c] for c in t if c in C2I2]
        if not e:
            e = [C2I2['0']]
        tgt.extend(e); lens.append(len(e))
    return torch.tensor(tgt, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def decode2(logits, ws=None):
    lp = logits.log_softmax(-1)
    p, idx = lp.max(-1)
    idx = idx.cpu().numpy(); p = p.cpu().numpy()
    outs, confs = [], []
    for b in range(idx.shape[0]):
        T = idx.shape[1]
        if ws is not None:
            T = max(1, min(T, int(math.ceil(ws[b] / 4))))
        seq = idx[b, :T]; pp = p[b, :T]
        s, prev = [], 0
        for j in range(T):
            k = int(seq[j])
            if k != prev and k != 0:
                s.append(I2C2[k])
            prev = k
        outs.append(''.join(s))
        confs.append(float(pp[:T].mean()))
    return outs, confs


def train(epochs=7, bs=96, lr=2e-3, ckpt='crnn2.pt'):
    with open(W + r'\cache\synth2.pkl', 'rb') as f:
        X, Y = pickle.load(f)
    n = len(X)
    order = np.argsort([x.shape[1] for x in X])
    net = CRNN(ncls=NCLS2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = n // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time()
    for ep in range(epochs):
        buckets = [order[i * bs:(i + 1) * bs] for i in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0.0
        for idx in buckets:
            xb, ws = pad_batch([X[i] for i in idx])
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
        print(f'ep {ep} loss {tot/steps:.4f} {time.time()-t0:.0f}s', flush=True)
        torch.save(net.state_dict(), W + '\\cache\\' + ckpt)
    net.eval()
    idx = list(range(n - 512, n))
    with torch.no_grad(), torch.amp.autocast('cuda'):
        xb, ws = pad_batch([X[i] for i in idx])
        lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
    outs, _ = decode2(lg.float(), ws)
    gt = [Y[i] for i in idx]
    print('holdout exact', np.mean([a == b for a, b in zip(outs, gt)]), flush=True)
    print(list(zip(outs[:12], gt[:12])))


if __name__ == '__main__':
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 7)
