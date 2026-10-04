"""Decode cached token strips with the extended-alphabet CRNN."""
import os, pickle, sys, time, math
import numpy as np, torch
from crnn import CRNN, pad_batch, dev
from crnn2 import decode2
from synth2 import NCLS2

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def run(net, split, bs=256):
    T = pickle.load(open(W + f'\\cache\\tok_{split}.pkl', 'rb'))
    flat, = [],
    flat = [s for strips, _ in T for s in strips]
    order = np.argsort([s.shape[1] for s in flat])
    texts = [''] * len(flat); confs = np.zeros(len(flat), np.float32)
    net.eval(); t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            xb, ws = pad_batch([flat[i] for i in idx])
            lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
            o, c = decode2(lg.float(), ws)
            for j, i in enumerate(idx):
                texts[i] = o[j]; confs[i] = c[j]
    print(split, len(flat), 'strips in', round(time.time() - t0, 1), 's', flush=True)
    out, p = [], 0
    for strips, feats in T:
        n = len(strips)
        out.append((texts[p:p + n], confs[p:p + n], feats)); p += n
    pickle.dump(out, open(W + f'\\cache\\dec_{split}.pkl', 'wb'), protocol=4)


if __name__ == '__main__':
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + r'\cache\crnn2.pt', map_location=dev))
    for s in sys.argv[1:] or ['train', 'test']:
        run(net, s)
