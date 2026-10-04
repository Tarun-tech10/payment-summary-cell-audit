"""CRNN + CTC digit-string reader for receipt number strips."""
import os, pickle, time, math, random
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
ALPHA = '0123456789,.-'
C2I = {c: i + 1 for i, c in enumerate(ALPHA)}
I2C = {i + 1: c for i, c in enumerate(ALPHA)}
NCLS = len(ALPHA) + 1
SH = 32
dev = 'cuda'


class CRNN(nn.Module):
    def __init__(s, ncls=NCLS, nh=256):
        super().__init__()
        def cbr(i, o, k=3, st=1, p=1):
            return nn.Sequential(nn.Conv2d(i, o, k, st, p, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(True))
        s.cnn = nn.Sequential(
            cbr(1, 48), cbr(48, 48), nn.MaxPool2d(2, 2),        # 16 x W/2
            cbr(48, 96), cbr(96, 96), nn.MaxPool2d(2, 2),       # 8 x W/4
            cbr(96, 160), cbr(160, 160), nn.MaxPool2d((2, 1)),  # 4 x W/4
            cbr(160, 224), nn.MaxPool2d((2, 1)),                # 2 x W/4
            cbr(224, 256, k=(2, 3), st=1, p=(0, 1)),            # 1 x W/4
        )
        s.rnn = nn.LSTM(256, nh, num_layers=2, bidirectional=True,
                        batch_first=True, dropout=0.15)
        s.fc = nn.Linear(2 * nh, ncls)
        s.drop = nn.Dropout(0.15)

    def forward(s, x):
        f = s.cnn(x)                       # B,C,1,T
        f = f.squeeze(2).permute(0, 2, 1)  # B,T,C
        f, _ = s.rnn(f)
        return s.fc(s.drop(f))             # B,T,ncls


def pad_batch(imgs, maxw=None):
    ws = [im.shape[1] for im in imgs]
    Wm = maxw or max(ws)
    Wm = int(math.ceil(Wm / 4) * 4)
    out = np.full((len(imgs), SH, Wm), 255, np.uint8)
    for i, im in enumerate(imgs):
        w = min(im.shape[1], Wm)
        out[i, :, :w] = im[:, :w]
    return out, ws


def encode(txts):
    tgt, lens = [], []
    for t in txts:
        e = [C2I[c] for c in t if c in C2I]
        tgt.extend(e); lens.append(len(e))
    return torch.tensor(tgt, dtype=torch.long), torch.tensor(lens, dtype=torch.long)


def greedy_decode(logits, ws=None):
    """logits B,T,C -> list of strings (with per-string mean prob)."""
    lp = logits.log_softmax(-1)
    p, idx = lp.max(-1)
    outs, confs = [], []
    idx = idx.cpu().numpy(); p = p.cpu().numpy()
    for b in range(idx.shape[0]):
        T = idx.shape[1]
        if ws is not None:
            T = max(1, min(T, int(math.ceil(ws[b] / 4))))
        seq = idx[b, :T]; pp = p[b, :T]
        s, prev, cp = [], 0, []
        for j in range(T):
            k = seq[j]
            if k != prev and k != 0:
                s.append(I2C[int(k)])
            prev = k
            cp.append(pp[j])
        outs.append(''.join(s))
        confs.append(float(np.mean(cp)) if cp else -9.9)
    return outs, confs


def train_synth(epochs=6, bs=96, lr=2e-3, limit=None, ckpt='crnn_synth.pt'):
    with open(W + r'\cache\synth.pkl', 'rb') as f:
        X, Y = pickle.load(f)
    if limit:
        X, Y = X[:limit], Y[:limit]
    n = len(X)
    order = np.argsort([x.shape[1] for x in X])  # bucket by width
    net = CRNN().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = n // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    t0 = time.time()
    for ep in range(epochs):
        buckets = [order[i * bs:(i + 1) * bs] for i in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0
        for bi, idx in enumerate(buckets):
            imgs = [X[i] for i in idx]; txts = [Y[i] for i in idx]
            xb, ws = pad_batch(imgs)
            xb = torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)
            tgt, tl = encode(txts)
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
    # quick sanity on held-out tail
    net.eval()
    idx = list(range(n - 256, n))
    with torch.no_grad(), torch.amp.autocast('cuda'):
        xb, ws = pad_batch([X[i] for i in idx])
        lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
    outs, _ = greedy_decode(lg.float(), ws)
    acc = np.mean([o == Y[i] for o, i in zip(outs, idx)])
    print('synthetic holdout exact acc', acc)
    print(list(zip(outs[:10], [Y[i] for i in idx[:10]])))
    return net


if __name__ == '__main__':
    import sys
    train_synth(epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
