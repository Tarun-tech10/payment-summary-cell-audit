"""Layout network with EXACT-label augmentation.

Translating the image by a whole grid cell, or mirroring it, transforms the 4x4 label in a
known way, so these are exact extra supervision rather than approximations -- worth a lot on
a 529-image training set.
"""
import os, json, math, time, sys
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score, fields_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
dev = 'cuda'
torch.backends.cudnn.benchmark = True
FKEYS = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
NI_MAX = 15


class Block(nn.Module):
    def __init__(s, ci, co, stride=2):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)

    def forward(s, x):
        x = F.relu(s.b1(s.c1(x)), True)
        return F.relu(s.b2(s.c2(x)), True)


class Net(nn.Module):
    def __init__(s, cin=5, wid=(32, 64, 96, 160, 224, 256), pdrop=0.3, deep=True):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(cin, wid[0], 5, 2, 2, bias=False),
                               nn.BatchNorm2d(wid[0]), nn.ReLU(True))
        s.b1 = Block(wid[0], wid[1]); s.b2 = Block(wid[1], wid[2])
        s.b3 = Block(wid[2], wid[3]); s.b4 = Block(wid[3], wid[4])
        s.b5 = Block(wid[4], wid[5]) if deep else None
        s.drop = nn.Dropout2d(pdrop)
        out = wid[5] if deep else wid[4]
        s.grid = nn.Conv2d(out, 4, 1)
        s.fc_ni = nn.Linear(out, NI_MAX + 1)
        s.fc_fp = nn.Linear(out, 4)

    def forward(s, x):
        x = s.b4(s.b3(s.b2(s.b1(s.stem(x)))))
        if s.b5 is not None:
            x = s.b5(x)
        x = s.drop(x)
        gl = s.grid(F.adaptive_avg_pool2d(x, (4, 4)))
        v = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return gl, s.fc_ni(v), s.fc_fp(v)


def photometric(x):
    B = x.shape[0]
    a = torch.rand(B, 1, 1, 1, device=x.device) * 0.5 + 0.75
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.25
    y = x.clone()
    y[:, :3] = (y[:, :3] * a + b).clamp(0, 1)
    y[:, 4:5] = (y[:, 4:5] * a + b).clamp(0, 1)
    y[:, :3] = (y[:, :3] * (1 + (torch.rand(B, 3, 1, 1, device=x.device) - .5) * .2)).clamp(0, 1)
    y = y + torch.randn_like(y) * 0.02
    ang = (torch.rand(B, device=x.device) - .5) * (2 * math.pi / 180 * 2.0)
    sc = 1 + (torch.rand(B, device=x.device) - .5) * 0.05
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    th = torch.zeros(B, 2, 3, device=x.device)
    th[:, 0, 0] = cos; th[:, 0, 1] = -sin
    th[:, 1, 0] = sin; th[:, 1, 1] = cos
    g = F.affine_grid(th, y.shape, align_corners=False)
    return F.grid_sample(y, g, padding_mode='border', align_corners=False)


def exact_shift(x, G, dx, dy, flip):
    """Whole-cell translate / mirror with the matching exact label transform.

    x: B,C,H,W (H,W divisible by 4).  G: B,4,4 long.  Returns (x', G')."""
    B, C, H, Wd = x.shape
    ch, cw = H // 4, Wd // 4
    if flip:
        x = torch.flip(x, dims=[3])
        G = torch.flip(G, dims=[2])
    if dx or dy:
        out = torch.zeros_like(x)
        sx, sy = dx * cw, dy * ch
        xs0, xs1 = max(0, sx), min(Wd, Wd + sx)
        xd0, xd1 = max(0, -sx), min(Wd, Wd - sx)
        ys0, ys1 = max(0, sy), min(H, H + sy)
        yd0, yd1 = max(0, -sy), min(H, H - sy)
        out[:, :, ys0:ys1, xs0:xs1] = x[:, :, yd0:yd1, xd0:xd1]
        x = out
        G2 = torch.zeros_like(G)
        r0, r1 = max(0, dy), min(4, 4 + dy)
        c0, c1 = max(0, dx), min(4, 4 + dx)
        G2[:, r0:r1, c0:c1] = G[:, max(0, -dy):min(4, 4 - dy), max(0, -dx):min(4, 4 - dx)]
        G = G2
    return x, G


def run_fold(Xtr, Ytr, seed, epochs, bs=16, lr=3e-3, wid=None, pdrop=0.3, deep=True,
             shift_p=0.5, flip_p=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(wid=wid or (32, 64, 96, 160, 224, 256), pdrop=pdrop, deep=deep).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    G, NI, FP = [t.to(dev) for t in Ytr]
    X = Xtr.to(dev)
    n = len(X); steps = max(1, n // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.25)
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = photometric(X[idx]); gb = G[idx]
            if rng.random() < shift_p:
                dx = int(rng.integers(-1, 2)); dy = int(rng.integers(-1, 2))
            else:
                dx = dy = 0
            fl = rng.random() < flip_p
            xb, gb = exact_shift(xb, gb, dx, dy, fl)
            with torch.amp.autocast('cuda'):
                gl, nl, fl_ = net(xb)
                loss = (F.cross_entropy(gl, gb, label_smoothing=0.03)
                        + 0.5 * F.cross_entropy(nl, NI[idx], label_smoothing=0.05)
                        + 0.5 * F.binary_cross_entropy_with_logits(fl_, FP[idx]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
    return net


def predict(net, X, bs=16):
    net.eval(); outs = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            acc = None
            for sc in (1.0, 1.03, 0.97):
                if sc == 1.0:
                    z = xb
                else:
                    th = torch.zeros(xb.shape[0], 2, 3, device=dev)
                    th[:, 0, 0] = 1 / sc; th[:, 1, 1] = 1 / sc
                    z = F.grid_sample(xb, F.affine_grid(th, xb.shape, align_corners=False),
                                      padding_mode='border', align_corners=False)
                gl, nl, fl = net(z)
                o = (gl.float().softmax(1), nl.float().softmax(1), torch.sigmoid(fl.float()))
                acc = o if acc is None else tuple(p + q for p, q in zip(acc, o))
            outs.append(tuple(p.cpu().numpy() / 3 for p in acc))
    return [np.concatenate([o[k] for o in outs]) for k in range(3)]


def main(tag, hires, epochs, seed0, wid, pdrop, deep, bs):
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    sfx = '_hi' if hires else ''
    Xi = np.load(W + f'\\cache\\train_img{sfx}.npy')
    Xe = np.load(W + f'\\cache\\test_img{sfx}.npy')
    X = torch.from_numpy(Xi).float().permute(0, 3, 1, 2) / 255.
    Xte = torch.from_numpy(Xe).float().permute(0, 3, 1, 2) / 255.
    G = torch.from_numpy(np.array([a['semantic_grid'] for a in A], np.int64).reshape(-1, 4, 4))
    NI = torch.from_numpy(np.array([min(a['num_items'], NI_MAX) for a in A], np.int64))
    FP = torch.from_numpy(np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32))
    ntr, nte = len(X), len(Xte)
    og = np.zeros((ntr, 4, 4, 4), np.float32); tg = np.zeros((nte, 4, 4, 4), np.float32)
    on = np.zeros((ntr, NI_MAX + 1), np.float32); tn = np.zeros((nte, NI_MAX + 1), np.float32)
    of = np.zeros((ntr, 4), np.float32); tf = np.zeros((nte, 4), np.float32)
    t0 = time.time()
    for k, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0)
                                 .split(np.zeros(ntr), np.minimum(NI.numpy(), 8))):
        net = run_fold(X[ti], (G[ti], NI[ti], FP[ti]), seed0 + k, epochs,
                       wid=wid, pdrop=pdrop, deep=deep, bs=bs)
        a, b, c = predict(net, X[vi]); og[vi] = a; on[vi] = b; of[vi] = c
        a, b, c = predict(net, Xte); tg += a / 5; tn += b / 5; tf += c / 5
        print(f'fold {k} {time.time()-t0:.0f}s', flush=True)
        del net; torch.cuda.empty_cache()
    np.save(W + f'\\cache\\oof{tag}_g.npy', og); np.save(W + f'\\cache\\te{tag}_g.npy', tg)
    np.save(W + f'\\cache\\oof{tag}_n.npy', on); np.save(W + f'\\cache\\te{tag}_n.npy', tn)
    np.save(W + f'\\cache\\oof{tag}_f.npy', of); np.save(W + f'\\cache\\te{tag}_f.npy', tf)
    Gn = np.array([a['semantic_grid'] for a in A])
    p = og.argmax(1).reshape(ntr, 16)
    gs = float(np.mean([grid_score(p[i], Gn[i]) for i in range(ntr)]))
    ns = float((on.argmax(1) == NI.numpy()).mean())
    pf = of > .5
    fs = float(np.mean([fields_score(dict(zip(FKEYS, pf[i])),
                                     dict(zip(FKEYS, FP.numpy()[i] > .5))) for i in range(ntr)]))
    print(f'[{tag}] GRID {gs:.4f} NUMITEMS {ns:.4f} FIELDS {fs:.4f}', flush=True)


if __name__ == '__main__':
    tag = sys.argv[1]
    hires = int(sys.argv[2])
    epochs = int(sys.argv[3])
    seed0 = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    deep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    bs = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    main(tag, hires, epochs, seed0, None, 0.3, bool(deep), bs)
