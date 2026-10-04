"""Decode a strip cache with a checkpoint and report the recall ceilings that matter.

Usage: python eval_reader.py <ckpt> [tokcache=tok3] [split=train] [save_tag]
"""
import os, sys, json, pickle, re, time
from collections import Counter
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from crnn2 import decode2
from synth2 import NCLS2
from reformat_test import variants, receipt_sep, dg

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
NUM = re.compile(r'-?\d[\d.,]*\d|\d')


def decode_all(net, tok, bs=96):
    flat = [s for strips, _ in tok for s in strips]
    order = np.argsort([s.shape[1] for s in flat])
    texts = [''] * len(flat); confs = np.zeros(len(flat), np.float32)
    net.eval(); t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for k in range(0, len(order), bs):
            ix = order[k:k + bs]
            xb, ws = pad_batch([flat[i] for i in ix])
            lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
            o, c = decode2(lg.float(), ws)
            for j, i in enumerate(ix):
                texts[i] = o[j]; confs[i] = c[j]
    print(f'  decoded {len(flat)} strips in {time.time()-t0:.0f}s', flush=True)
    out, p = [], 0
    for strips, F in tok:
        n = len(strips)
        out.append((texts[p:p + n], confs[p:p + n], F)); p += n
    return out


def report(dec, golds):
    n = len(golds)
    raw = ref = digok = 0
    for i in range(n):
        texts = dec[i][0]
        cands, digits = set(), set()
        for t in texts:
            t = t.strip()
            if t:
                cands.add(t)
                if dg(t):
                    digits.add(dg(t))
            for m in NUM.findall(t):
                m = m.strip('.,')
                if m:
                    cands.add(m)
                    if dg(m):
                        digits.add(dg(m))
        sep = receipt_sep(texts)
        rf = set()
        for d in digits:
            if 2 <= len(d) <= 9:
                rf |= variants(d, sep)
        raw += golds[i] in cands
        ref += golds[i] in (cands | rf)
        digok += dg(golds[i]) in digits
    return raw / n, ref / n, digok / n


if __name__ == '__main__':
    ck = sys.argv[1]
    cache = sys.argv[2] if len(sys.argv) > 2 else 'tok3'
    split = sys.argv[3] if len(sys.argv) > 3 else 'train'
    tag = sys.argv[4] if len(sys.argv) > 4 else None
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    tok = pickle.load(open(W + f'\\cache\\{cache}_{split}.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    dec = decode_all(net, tok)
    if tag:
        pickle.dump(dec, open(W + f'\\cache\\{tag}_{split}.pkl', 'wb'), protocol=4)
    if split == 'train':
        r, f, d = report(dec, golds)
        print(f'{ck} on {cache}: raw {r:.4f}  +reformat {f:.4f}  digits {d:.4f}', flush=True)
