"""Longer / multi-seed version with TTA; saves OOF + test predictions."""
import os, json, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score, fields_score
from cnn import Net, load, to_tensor, run_fold, FKEYS, NI_MAX, D, W, dev

EPOCHS = 130
SEEDS = [0]

def tta_predict(net, X, bs=32):
    net.eval()
    outs = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(dev)
            acc = None
            for sc in [1.0, 1.03, 0.97]:
                if sc == 1.0:
                    z = xb
                else:
                    B = xb.shape[0]
                    th = torch.zeros(B, 2, 3, device=dev)
                    th[:, 0, 0] = 1 / sc; th[:, 1, 1] = 1 / sc
                    g = F.affine_grid(th, xb.shape, align_corners=False)
                    z = F.grid_sample(xb, g, padding_mode='border', align_corners=False)
                gl, nl, fl = net(z)
                o = (gl.float().softmax(1), nl.float().softmax(1), torch.sigmoid(fl.float()))
                acc = o if acc is None else tuple(a + b for a, b in zip(acc, o))
            outs.append(tuple(a.cpu() / 3 for a in acc))
    return [torch.cat([o[k] for o in outs]) for k in range(3)]


if __name__ == '__main__':
    tr, A, Xi, G, NI, FP = load()
    te = pd.read_csv(D + r'\test.csv')
    Xte = to_tensor(np.load(W + r'\cache\test_img.npy'))
    X = to_tensor(Xi)
    Gt = torch.from_numpy(G); NIt = torch.from_numpy(NI); FPt = torch.from_numpy(FP)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    strat = np.minimum(NI, 8)
    oof_g = np.zeros((len(X), 4, 4, 4), np.float32)
    oof_n = np.zeros((len(X), NI_MAX + 1), np.float32)
    oof_f = np.zeros((len(X), 4), np.float32)
    te_g = np.zeros((len(Xte), 4, 4, 4), np.float32)
    te_n = np.zeros((len(Xte), NI_MAX + 1), np.float32)
    te_f = np.zeros((len(Xte), 4), np.float32)
    t0 = time.time(); nrun = 0
    for seed in SEEDS:
        for k, (ti, vi) in enumerate(skf.split(np.zeros(len(X)), strat)):
            net, _ = run_fold(X[ti], (Gt[ti], NIt[ti], FPt[ti]), X[vi][:1],
                              seed=seed * 100 + k, epochs=EPOCHS)
            pg, pn, pf = tta_predict(net, X[vi])
            oof_g[vi] += pg.numpy() / len(SEEDS)
            oof_n[vi] += pn.numpy() / len(SEEDS)
            oof_f[vi] += pf.numpy() / len(SEEDS)
            qg, qn, qf = tta_predict(net, Xte)
            te_g += qg.numpy() / (5 * len(SEEDS))
            te_n += qn.numpy() / (5 * len(SEEDS))
            te_f += qf.numpy() / (5 * len(SEEDS))
            torch.save(net.state_dict(), W + f'\\cache\\cnn2_s{seed}_f{k}.pt')
            nrun += 1
            print('run', nrun, 'elapsed', round(time.time() - t0, 1), flush=True)
    np.save(W + r'\cache\oof2_g.npy', oof_g); np.save(W + r'\cache\oof2_n.npy', oof_n)
    np.save(W + r'\cache\oof2_f.npy', oof_f)
    np.save(W + r'\cache\te2_g.npy', te_g); np.save(W + r'\cache\te2_n.npy', te_n)
    np.save(W + r'\cache\te2_f.npy', te_f)
    pg = oof_g.argmax(1)
    gs = np.mean([grid_score(pg[i].ravel(), G[i].ravel()) for i in range(len(G))])
    ns = np.mean(oof_n.argmax(1) == NI)
    pf = (oof_f > 0.5)
    fs = np.mean([fields_score(dict(zip(FKEYS, pf[i])), dict(zip(FKEYS, FP[i] > .5))) for i in range(len(FP))])
    print(f'GRID {gs:.4f}  NUMITEMS {ns:.4f}  FIELDS {fs:.4f}')
    print('partial', 0.45 * gs + 0.12 * ns + 0.08 * fs)
