"""N-best readings for the strips that could plausibly hold the total.

99.8% of receipts contain a crop whose CTC loss against the gold total is under 0.2 -- the
correct string is a high-probability path that greedy decoding simply does not take.  Rather
than a full prefix beam over every strip (far too slow here), this flips the single most
uncertain frames of the greedy path, which is where a one-character misread lives, and only
for the strips near the summary block.  Cheap, and aimed exactly at the recall ceiling.

Writes cache/beamB_<split>.pkl in the format final2.cands_for_image already consumes.
"""
import os, sys, pickle, time
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from synth2 import NCLS2, I2C2
from final2 import dg, C_P2, C_P3

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def collapse(seq):
    out, prev = [], 0
    for k in seq:
        k = int(k)
        if k != prev and k != 0:
            out.append(I2C2.get(k, ''))
        prev = k
    return ''.join(out)


def pick_strips(F, M):
    """The strips most likely to sit in the subtotal/total block."""
    if len(F) == 0:
        return np.zeros(0, int)
    s = F[:, C_P2] + F[:, C_P3]
    return np.argsort(-s)[:M]


def run(ck, split, M=44, R=7, bs=64, out='beamB'):
    tok = pickle.load(open(W + f'\\cache\\tok3_{split}.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    net.eval()
    beams, t0 = [], time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, (strips, F) in enumerate(tok):
            sel = pick_strips(F, M)
            per = []
            for k in range(0, len(sel), bs):
                ix = sel[k:k + bs]
                xb, ws = pad_batch([strips[j] for j in ix])
                lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
                lp = lg.float().log_softmax(-1)
                v, idx = lp.topk(2, -1)
                v = v.cpu().numpy(); idx = idx.cpu().numpy()
                for b, j in enumerate(ix):
                    T = max(1, min(lp.shape[1], int(np.ceil(ws[b] / 4))))
                    path = idx[b, :T, 0].copy()
                    marg = v[b, :T, 0] - v[b, :T, 1]
                    base = collapse(path)
                    seen = {base}
                    alts = []
                    for j2 in np.argsort(marg)[:R]:
                        p2 = path.copy(); p2[j2] = idx[b, j2, 1]
                        s = collapse(p2)
                        if s and s not in seen and 2 <= len(dg(s)) <= 9:
                            seen.add(s)
                            alts.append((s, float(-marg[j2])))
                    if alts:
                        per.append((int(j), alts))
            beams.append(per)
            if i % 120 == 0:
                print(f'  {split} {i}/{len(tok)} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(beams, open(W + f'\\cache\\{out}_{split}.pkl', 'wb'), protocol=4)
    n = float(np.mean([sum(len(a) for _, a in b) for b in beams]))
    print(f'{split}: {n:.1f} alternate readings per receipt ({time.time()-t0:.0f}s)', flush=True)
    return beams


if __name__ == '__main__':
    ck = sys.argv[1] if len(sys.argv) > 1 else 'crnn3.pt'
    for sp in ('train', 'test'):
        run(ck, sp)
