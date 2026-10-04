"""Money-constrained beam decoding of the most promising strips per image.

Adds extra candidate strings (with their scores) on top of the greedy decodes.
Usage: python beam_decode.py <ckpt> <tag> <split> [topn_per_image]
"""
import os, sys, json, pickle, time
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from synth2 import NCLS2, ALPHA2
from beam import prefix_beam, build_prior, rescore, MONEY

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
I2C2 = {i + 1: c for i, c in enumerate(ALPHA2)}
ALLOWED = [i + 1 for i, c in enumerate(ALPHA2) if c in MONEY]
C_P3, C_P2, C_KIND, C_RIGHT = 15, 14, 6, 11


def pick_strips(F, dec_texts, n=60):
    """Strips worth an expensive decode: total/subtotal area, right-aligned, digit-bearing."""
    if len(F) == 0:
        return []
    import re
    score = (F[:, C_P3] * 2.0 + F[:, C_P2] + 0.5 * F[:, C_RIGHT]
             + np.array([0.8 if len(re.sub(r'\D', '', t)) >= 3 else 0.0 for t in dec_texts]))
    return list(np.argsort(-score)[:n])


def main(ckpt, tag, split, topn=60, bs=192):
    tr = pd.read_csv(D + r'\train.csv')
    prior, dflt = build_prior([json.loads(x)['total_price'] for x in tr.answer_json])
    tok = pickle.load(open(W + f'\\cache\\tok2_{split}.pkl', 'rb'))
    dec = pickle.load(open(W + f'\\cache\\dec2B_{split}.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ckpt, map_location=dev))
    net.eval()
    out = []
    t0 = time.time()
    for i, ((strips, F), (texts, confs, _)) in enumerate(zip(tok, dec)):
        sel = pick_strips(F, texts, topn)
        res = []
        for k in range(0, len(sel), bs):
            ix = sel[k:k + bs]
            xb, ws = pad_batch([strips[j] for j in ix])
            with torch.no_grad(), torch.amp.autocast('cuda'):
                lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)).float()
            lp = lg.log_softmax(-1).cpu().numpy()
            for a, j in enumerate(ix):
                T = max(1, min(lp.shape[1], int(np.ceil(ws[a] / 4))))
                cands = prefix_beam(lp[a, :T], ALLOWED, I2C2, beam=10, topk=4)
                if cands:
                    res.append((int(j), rescore(cands, prior, dflt)))
        out.append(res)
        if i % 100 == 0:
            print(f'{split} {i} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(out, open(W + f'\\cache\\beam{tag}_{split}.pkl', 'wb'), protocol=4)
    print('done', split, round(time.time() - t0, 1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 60)
