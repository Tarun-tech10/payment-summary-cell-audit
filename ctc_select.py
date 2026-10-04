"""Pick, for each training image, the strip whose CTC loss under the gold string is lowest.

Far more sensitive than exact-match decoding: it finds the correct crop even when the
greedy decode is wrong, which gives us real (crop, label) pairs to adapt the reader on.
"""
import os, pickle, json, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from crnn import CRNN, pad_batch, dev
from crnn2 import encode2, decode2
from synth2 import NCLS2, C2I2

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def gold_losses(net, strips, gold, bs=256):
    """Per-strip CTC loss of the gold label (lower = this crop probably is the total)."""
    if not strips:
        return np.zeros(0, np.float32)
    tgt = torch.tensor([C2I2[c] for c in gold if c in C2I2], dtype=torch.long)
    if len(tgt) == 0:
        return np.full(len(strips), 99.0, np.float32)
    ctc = nn.CTCLoss(blank=0, reduction='none', zero_infinity=True)
    out = np.zeros(len(strips), np.float32)
    order = np.argsort([s.shape[1] for s in strips])
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            xb, ws = pad_batch([strips[i] for i in idx])
            lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)).float()
            lp = lg.log_softmax(-1).permute(1, 0, 2)
            n = len(idx)
            T = torch.tensor([max(1, min(lp.shape[0], int(np.ceil(ws[j] / 4)))) for j in range(n)])
            tg = tgt.repeat(n)
            tl = torch.full((n,), len(tgt), dtype=torch.long)
            loss = ctc(lp, tg, T, tl).cpu().numpy() / max(len(tgt), 1)
            for j, i in enumerate(idx):
                out[i] = loss[j]
    return out


def main(ckpt='crnn_ft.pt', out='real_pairs.pkl', topk=2, thr=1.2):
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    tok = pickle.load(open(W + r'\cache\tok2_train.pkl', 'rb'))
    dec = pickle.load(open(W + r'\cache\dec2_train.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ckpt, map_location=dev))
    net.eval()
    pairs, stats, sel_idx = [], [], []
    t0 = time.time()
    for i, ((strips, F), g) in enumerate(zip(tok, golds)):
        L = gold_losses(net, strips, g)
        if len(L) == 0:
            stats.append((99.0, False)); sel_idx.append([]); continue
        o = np.argsort(L)[:topk]
        keep = [int(j) for j in o if L[j] < thr]
        sel_idx.append(keep)
        exact = any(dec[i][0][j].strip() == g for j in range(len(strips)))
        stats.append((float(L[o[0]]), exact))
        for j in keep:
            pairs.append((strips[j], g))
    S = np.array([s[0] for s in stats]); E = np.array([s[1] for s in stats])
    print(f'{len(pairs)} real pairs from {sum(1 for k in sel_idx if k)} images '
          f'({time.time()-t0:.0f}s)', flush=True)
    for t in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6]:
        m = S < t
        print(f'  loss<{t}: {m.mean():.3f} of images   (exact-decode rate there {E[m].mean() if m.any() else 0:.3f})')
    print('exact-decode recall overall', E.mean())
    pickle.dump((pairs, sel_idx, S), open(W + '\\cache\\' + out, 'wb'), protocol=4)


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
