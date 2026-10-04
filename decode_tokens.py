"""Run the CRNN over all cached token strips; cache decoded strings + confidences."""
import os, pickle, sys, time, math
import numpy as np, torch
from crnn import CRNN, pad_batch, greedy_decode, dev, SH

W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'


def decode_split(net, split, bs=256):
    T = pickle.load(open(W + f'\\cache\\tok_{split}.pkl', 'rb'))
    flat, owner = [], []
    for i, (strips, feats) in enumerate(T):
        for s in strips:
            flat.append(s); owner.append(i)
    order = np.argsort([s.shape[1] for s in flat])
    texts = [None] * len(flat); confs = [0.0] * len(flat)
    net.eval()
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for k in range(0, len(order), bs):
            idx = order[k:k + bs]
            xb, ws = pad_batch([flat[i] for i in idx])
            lg = net(torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev))
            o, c = greedy_decode(lg.float(), ws)
            for j, i in enumerate(idx):
                texts[i] = o[j]; confs[i] = c[j]
    print(split, len(flat), 'strips decoded in', round(time.time() - t0, 1), 's', flush=True)
    out = []
    p = 0
    for i, (strips, feats) in enumerate(T):
        n = len(strips)
        out.append((texts[p:p + n], np.array(confs[p:p + n], np.float32), feats))
        p += n
    with open(W + f'\\cache\\dec_{split}.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return out


if __name__ == '__main__':
    ck = sys.argv[1] if len(sys.argv) > 1 else 'crnn_synth.pt'
    net = CRNN().to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + ck, map_location=dev))
    for split in sys.argv[2:] or ['train']:
        decode_split(net, split)
