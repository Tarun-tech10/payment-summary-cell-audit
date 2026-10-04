"""Width-scale TTA decoding: read every strip at several horizontal scales.

CTC output is sensitive to horizontal scale, and candidate recall is a union over
readings, so each scale is another chance at the right digits.
Usage: python decode_tta.py <ckpt> <out_tag> <split> [scales]
"""
import os, sys, json, pickle, time, gc
import numpy as np, pandas as pd, cv2, torch
from crnn import CRNN, pad_batch, dev
from crnn2 import decode2
from synth2 import NCLS2
from eval_reader import report

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def rescale(im, f):
    if f == 1.0:
        return im
    w = max(8, int(round(im.shape[1] * f)))
    return cv2.resize(im, (w, im.shape[0]), interpolation=cv2.INTER_LINEAR)


def run(ck, tag, split, scales=(1.0, 0.88, 1.14), bs=96):
    tok = pickle.load(open(W + f'\\cache\\tok3_{split}.pkl', 'rb'))
    counts = [len(s) for s, _ in tok]
    flat = [s for strips, _ in tok for s in strips]
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    net.eval()
    per_scale = []
    t0 = time.time()
    for f in scales:
        sc = [rescale(s, f) for s in flat]
        order = np.argsort([s.shape[1] for s in sc])
        texts = [''] * len(sc); confs = np.zeros(len(sc), np.float32)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for k in range(0, len(order), bs):
                ix = order[k:k + bs]
                xb, ws = pad_batch([sc[i] for i in ix])
                lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
                o, c = decode2(lg.float(), ws)
                for j, i in enumerate(ix):
                    texts[i] = o[j]; confs[i] = c[j]
        per_scale.append((texts, confs))
        del sc; gc.collect(); torch.cuda.empty_cache()
        print(f'  scale {f} done ({time.time()-t0:.0f}s)', flush=True)
    out, p = [], 0
    for (strips, Fm), n in zip(tok, counts):
        t = [x for ts, _ in per_scale for x in ts[p:p + n]]
        c = np.concatenate([cs[p:p + n] for _, cs in per_scale])
        out.append((t, c, np.tile(Fm, (len(scales), 1))))
        p += n
    pickle.dump(out, open(W + f'\\cache\\{tag}_{split}.pkl', 'wb'), protocol=4)
    if split == 'train':
        tr = pd.read_csv(D + r'\train.csv')
        golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
        r, rf, d = report(out, golds)
        print(f'{ck} TTA {scales}: raw {r:.4f}  +reformat {rf:.4f}  digits {d:.4f}', flush=True)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])
