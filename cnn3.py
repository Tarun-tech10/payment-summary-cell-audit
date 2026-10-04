"""Third layout-network run (different seeds/width) for grid ensembling."""
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from scorer import grid_score
from cnn import load, to_tensor, run_fold, NI_MAX, D, W
from cnn2 import tta_predict
import cnn

# slightly different capacity so the runs decorrelate
_orig = cnn.Net
class Net3(_orig):
    def __init__(s):
        super().__init__(wid=(40, 72, 112, 176, 256), pdrop=0.35)
cnn.Net = Net3

if __name__ == '__main__':
    tr, A, Xi, G, NI, FP = load()
    te = pd.read_csv(D + r'\test.csv')
    X = to_tensor(Xi); Xte = to_tensor(np.load(W + r'\cache\test_img.npy'))
    Gt = torch.from_numpy(G); NIt = torch.from_numpy(NI); FPt = torch.from_numpy(FP)
    og = np.zeros((len(X), 4, 4, 4), np.float32); tg = np.zeros((len(Xte), 4, 4, 4), np.float32)
    on = np.zeros((len(X), NI_MAX + 1), np.float32); tn = np.zeros((len(Xte), NI_MAX + 1), np.float32)
    of = np.zeros((len(X), 4), np.float32); tf = np.zeros((len(Xte), 4), np.float32)
    for k, (ti, vi) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0)
                                 .split(np.zeros(len(X)), np.minimum(NI, 8))):
        net, _ = run_fold(X[ti], (Gt[ti], NIt[ti], FPt[ti]), X[vi][:1], seed=200 + k, epochs=130)
        a, b, c = tta_predict(net, X[vi])
        og[vi] = a.numpy(); on[vi] = b.numpy(); of[vi] = c.numpy()
        a, b, c = tta_predict(net, Xte)
        tg += a.numpy() / 5; tn += b.numpy() / 5; tf += c.numpy() / 5
        print('fold', k, flush=True)
    np.save(W + r'\cache\oof3_g.npy', og); np.save(W + r'\cache\te3_g.npy', tg)
    np.save(W + r'\cache\oof3_n.npy', on); np.save(W + r'\cache\te3_n.npy', tn)
    np.save(W + r'\cache\oof3_f.npy', of); np.save(W + r'\cache\te3_f.npy', tf)
    p = og.argmax(1).reshape(len(G), 16)
    print('grid oof (run3)', np.mean([grid_score(p[i], G[i]) for i in range(len(G))]))
