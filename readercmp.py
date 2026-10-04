"""Is the self-trained reader actually better?  Decode a subset with each and compare.

Cheap gate before spending an hour of contended GPU on a full re-decode: what matters is
whether the correct digit string appears among a receipt's candidates at all.
"""
import sys, pickle, json, time
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from crnn2 import decode2
from synth2 import NCLS2
from final2 import dg, reformats, receipt_sep, NUM

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def decode_subset(ck, tok, idx, bs=96):
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    net.eval()
    out = []
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in idx:
            strips = tok[i][0]
            if not strips:
                out.append([]); continue
            order = np.argsort([s.shape[1] for s in strips])
            texts = [''] * len(strips)
            for k in range(0, len(order), bs):
                ix = order[k:k + bs]
                xb, ws = pad_batch([strips[j] for j in ix])
                lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
                o, _ = decode2(lg.float(), ws)
                for j, t in zip(ix, o):
                    texts[j] = t
            out.append(texts)
    del net; torch.cuda.empty_cache()
    return out


def recall(reads, golds, idx):
    hs = hd = 0
    for k, i in enumerate(idx):
        texts = reads[k]
        sep = receipt_sep(texts)
        pool = set()
        for t in texts:
            for m in [t.strip()] + NUM.findall(t):
                m = m.strip('.,'); d = dg(m)
                if 2 <= len(d) <= 9:
                    pool.add(m); pool |= reformats(d, sep)
        hs += golds[i] in pool
        hd += dg(golds[i]) in {dg(x) for x in pool}
    return hs / len(idx), hd / len(idx)


if __name__ == '__main__':
    nimg = int(sys.argv[1]) if len(sys.argv) > 1 else 140
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    tok = pickle.load(open(W + r'\cache\tok3_train.pkl', 'rb'))
    idx = np.linspace(0, len(tok) - 1, nimg).astype(int)
    print(f'{nimg} images, {sum(len(tok[i][0]) for i in idx)} strips', flush=True)
    for ck in ('crnn3.pt', 'crnn5.pt'):
        t0 = time.time()
        r = decode_subset(ck, tok, idx)
        s, d = recall(r, golds, idx)
        print(f'{ck:10s} exact {s:.4f}  digits {d:.4f}   ({time.time()-t0:.0f}s)', flush=True)
