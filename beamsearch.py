"""Proper CTC prefix beam search over the strips that could hold the total.

99.8% of receipts contain a crop whose CTC loss against the gold total is under 0.2, so the
correct string is a high-probability path the greedy argmax simply does not take.  Flipping
single frames only reaches one-character substitutions; a prefix beam also reaches
insertions, deletions and multi-character corrections, which is where the rest of the
recall ceiling is.

Run over only the top-M summary-area strips per receipt, so the cost is ~8k strips rather
than 265k and the whole thing fits in a couple of minutes.
"""
import os, sys, math, pickle, time
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from synth2 import NCLS2, I2C2
from final2 import dg, C_P2, C_P3

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
NEG = -1e30


def lse(a, b):
    if b > a:
        a, b = b, a
    if b <= -1e29:
        return a
    return a + math.log1p(math.exp(b - a))


def prefix_beam(lp, T, K=12, topn=6):
    """lp: (Tmax, C) log-probs. Returns [(string, logprob), ...] best first."""
    beams = {(): (0.0, NEG)}                      # prefix -> (ends-in-blank, ends-in-symbol)
    for t in range(T):
        row = lp[t]
        cand = np.argpartition(-row, topn)[:topn] if len(row) > topn else np.arange(len(row))
        nxt = {}
        for pref, (pb, pnb) in beams.items():
            ptot = lse(pb, pnb)
            last = pref[-1] if pref else -1
            for c in cand:
                p = float(row[c])
                if c == 0:                        # blank keeps the prefix
                    e = nxt.get(pref)
                    if e is None:
                        nxt[pref] = (ptot + p, NEG)
                    else:
                        nxt[pref] = (lse(e[0], ptot + p), e[1])
                elif c == last:
                    # repeat: collapses unless separated by a blank
                    e = nxt.get(pref)
                    if e is None:
                        nxt[pref] = (NEG, pnb + p)
                    else:
                        nxt[pref] = (e[0], lse(e[1], pnb + p))
                    ext = pref + (int(c),)
                    e2 = nxt.get(ext)
                    if e2 is None:
                        nxt[ext] = (NEG, pb + p)
                    else:
                        nxt[ext] = (e2[0], lse(e2[1], pb + p))
                else:
                    ext = pref + (int(c),)
                    e2 = nxt.get(ext)
                    if e2 is None:
                        nxt[ext] = (NEG, ptot + p)
                    else:
                        nxt[ext] = (e2[0], lse(e2[1], ptot + p))
        beams = dict(sorted(nxt.items(), key=lambda kv: -lse(kv[1][0], kv[1][1]))[:K])
    out = [(''.join(I2C2.get(c, '') for c in p), lse(v[0], v[1])) for p, v in beams.items()]
    out.sort(key=lambda x: -x[1])
    return out


def run(ck, split, M=16, K=12, topn=6, keep=6, bs=64, out='beamB'):
    tok = pickle.load(open(W + f'\\cache\\tok3_{split}.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    net.eval()
    beams, t0 = [], time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i, (strips, F) in enumerate(tok):
            per = []
            if len(strips):
                s = F[:, C_P2] + F[:, C_P3]
                sel = np.argsort(-s)[:M]
                for k in range(0, len(sel), bs):
                    ix = sel[k:k + bs]
                    xb, ws = pad_batch([strips[j] for j in ix])
                    lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
                    lp = lg.float().log_softmax(-1).cpu().numpy()
                    for b, j in enumerate(ix):
                        T = max(1, min(lp.shape[1], int(np.ceil(ws[b] / 4))))
                        cands = prefix_beam(lp[b], T, K=K, topn=topn)
                        alts = [(s2, float(v)) for s2, v in cands[1:]
                                if s2 and 2 <= len(dg(s2)) <= 9][:keep]
                        if alts:
                            per.append((int(j), alts))
            beams.append(per)
            if i % 100 == 0:
                print(f'  {split} {i}/{len(tok)} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(beams, open(W + f'\\cache\\{out}_{split}.pkl', 'wb'), protocol=4)
    n = float(np.mean([sum(len(a) for _, a in b) for b in beams]))
    print(f'{split}: {n:.1f} beam readings per receipt ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    ck = sys.argv[1] if len(sys.argv) > 1 else 'crnn3.pt'
    for sp in ('train', 'test'):
        run(ck, sp)
