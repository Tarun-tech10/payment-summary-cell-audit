"""Multi-task CNN: 4x4 semantic grid + num_items + fields_present."""
import os, json, math, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score, fields_score

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
dev = 'cuda'
torch.backends.cudnn.benchmark = True

FKEYS = ['subtotal_present', 'service_present', 'tax_present', 'etc_present']
NI_MAX = 15  # classes 0..15, >15 clipped


def load():
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    Xi = np.load(W + r'\cache\train_img.npy')
    G = np.array([a['semantic_grid'] for a in A], np.int64).reshape(-1, 4, 4)
    NI = np.array([min(a['num_items'], NI_MAX) for a in A], np.int64)
    FP = np.array([[a['fields_present'][k] for k in FKEYS] for a in A], np.float32)
    return tr, A, Xi, G, NI, FP


def to_tensor(X):
    """uint8 HWC(5) -> float CHW normalised."""
    x = torch.from_numpy(X).float().permute(0, 3, 1, 2) / 255.
    return x


class Block(nn.Module):
    def __init__(s, ci, co, stride=2):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)
    def forward(s, x):
        x = F.relu(s.b1(s.c1(x)), True)
        x = F.relu(s.b2(s.c2(x)), True)
        return x


class Net(nn.Module):
    def __init__(s, cin=5, wid=(32, 64, 96, 160, 224), pdrop=0.3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(cin, wid[0], 5, 2, 2, bias=False),
                               nn.BatchNorm2d(wid[0]), nn.ReLU(True))
        s.b1 = Block(wid[0], wid[1])
        s.b2 = Block(wid[1], wid[2])
        s.b3 = Block(wid[2], wid[3])
        s.b4 = Block(wid[3], wid[4])
        s.drop = nn.Dropout2d(pdrop)
        s.grid = nn.Conv2d(wid[4], 4, 1)
        s.fc_ni = nn.Linear(wid[4], NI_MAX + 1)
        s.fc_fp = nn.Linear(wid[4], 4)
    def forward(s, x):
        x = s.stem(x); x = s.b1(x); x = s.b2(x); f3 = s.b3(x); f = s.b4(f3)
        f = s.drop(f)
        gp = F.adaptive_avg_pool2d(f, (4, 4))
        gl = s.grid(gp)                      # B,4,4,4  (cls,row,col)
        v = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return gl, s.fc_ni(v), s.fc_fp(v)


def augment(x):
    B = x.shape[0]
    # brightness / contrast / gamma on the 3 photometric channels
    a = (torch.rand(B, 1, 1, 1, device=x.device) * 0.5 + 0.75)
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.25
    y = x.clone()
    y[:, :3] = (y[:, :3] * a + b).clamp(0, 1)
    y[:, 4:5] = (y[:, 4:5] * a + b).clamp(0, 1)
    # per-channel colour jitter
    y[:, :3] = (y[:, :3] * (1 + (torch.rand(B, 3, 1, 1, device=x.device) - .5) * .2)).clamp(0, 1)
    y = y + torch.randn_like(y) * 0.02
    # random small affine (scale/translate/rotate) -- labels approximately invariant
    ang = (torch.rand(B, device=x.device) - .5) * (2 * math.pi / 180 * 2.5)
    sc = 1 + (torch.rand(B, device=x.device) - .5) * 0.06
    tx = (torch.rand(B, device=x.device) - .5) * 0.04
    ty = (torch.rand(B, device=x.device) - .5) * 0.04
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, y.shape, align_corners=False)
    y = F.grid_sample(y, grid, padding_mode='border', align_corners=False)
    return y


def run_fold(Xtr, Ytr, Xva, seed=0, epochs=70, bs=16, lr=3e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    n = len(Xtr)
    steps = max(1, n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.25)
    G, NI, FP = Ytr
    scaler = torch.amp.GradScaler('cuda')
    Xtr_t = Xtr.to(dev); G = G.to(dev); NI = NI.to(dev); FP = FP.to(dev)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb = augment(Xtr_t[idx])
            with torch.amp.autocast('cuda'):
                gl, nl, fl = net(xb)
                loss = (F.cross_entropy(gl, G[idx], label_smoothing=0.03)
                        + 0.5 * F.cross_entropy(nl, NI[idx], label_smoothing=0.05)
                        + 0.5 * F.binary_cross_entropy_with_logits(fl, FP[idx]))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    net.eval()
    outs = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        Xva = Xva.to(dev)
        for i in range(0, len(Xva), 32):
            gl, nl, fl = net(Xva[i:i + 32])
            outs.append((gl.float().softmax(1).cpu(), nl.float().softmax(1).cpu(),
                         torch.sigmoid(fl.float()).cpu()))
    return net, [torch.cat([o[k] for o in outs]) for k in range(3)]


if __name__ == '__main__':
    tr, A, Xi, G, NI, FP = load()
    X = to_tensor(Xi)
    Gt = torch.from_numpy(G); NIt = torch.from_numpy(NI); FPt = torch.from_numpy(FP)
    nfold = 5
    skf = StratifiedKFold(nfold, shuffle=True, random_state=0)
    strat = np.minimum(NI, 8)
    oof_g = np.zeros((len(X), 4, 4, 4), np.float32)
    oof_n = np.zeros((len(X), NI_MAX + 1), np.float32)
    oof_f = np.zeros((len(X), 4), np.float32)
    t0 = time.time()
    for k, (ti, vi) in enumerate(skf.split(np.zeros(len(X)), strat)):
        net, (pg, pn, pf) = run_fold(X[ti], (Gt[ti], NIt[ti], FPt[ti]), X[vi], seed=k)
        oof_g[vi] = pg.numpy(); oof_n[vi] = pn.numpy(); oof_f[vi] = pf.numpy()
        torch.save(net.state_dict(), W + f'\\cache\\cnn_f{k}.pt')
        print('fold', k, 'done', round(time.time() - t0, 1), 's')
    np.save(W + r'\cache\oof_g.npy', oof_g)
    np.save(W + r'\cache\oof_n.npy', oof_n)
    np.save(W + r'\cache\oof_f.npy', oof_f)

    pg = oof_g.argmax(1)
    gs = np.mean([grid_score(pg[i].ravel(), G[i].ravel()) for i in range(len(G))])
    ns = np.mean(oof_n.argmax(1) == NI)
    pf = (oof_f > 0.5)
    fs = np.mean([fields_score(dict(zip(FKEYS, pf[i])), dict(zip(FKEYS, FP[i] > .5))) for i in range(len(FP))])
    print(f'GRID {gs:.4f}  NUMITEMS {ns:.4f}  FIELDS {fs:.4f}')
    print('partial score', 0.45 * gs + 0.12 * ns + 0.08 * fs)
