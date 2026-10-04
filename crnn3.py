"""Base reader trained on the wider synth3 corpus."""
import os, pickle, time, random, sys
import numpy as np, torch
import torch.nn as nn
from crnn import CRNN, pad_batch, dev
from crnn2 import encode2, decode2
from synth2 import NCLS2

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
LOGF = W + r'\progress_crnn3.txt'


def log(m):
    with open(LOGF, 'a') as f:
        f.write(str(m) + chr(10)); f.flush()
    print(m, flush=True)


def train(epochs=6, bs=112, lr=2e-3, ckpt='crnn3.pt'):
    X, Y = pickle.load(open(W + r'\cache\synth3.pkl', 'rb'))
    n = len(X)
    order = np.argsort([x.shape[1] for x in X])
    net = CRNN(ncls=NCLS2).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = n // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.15)
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
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
        log(f'ep {ep} loss {tot/steps:.4f} {time.time()-t0:.0f}s')
        torch.save(net.state_dict(), W + '\\cache\\' + ckpt)
    net.eval()
    idx = list(range(n - 512, n))
    with torch.no_grad(), torch.amp.autocast('cuda'):
        xb, ws = pad_batch([X[i] for i in idx])
        lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
    outs, _ = decode2(lg.float(), ws)
    log(f'holdout exact {np.mean([a == Y[i] for a, i in zip(outs, idx)]):.4f}')


if __name__ == '__main__':
    train(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
