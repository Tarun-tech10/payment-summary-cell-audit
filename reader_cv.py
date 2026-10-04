"""Fold-honest adaptation of the reader on real receipt crops.

For each fold we adapt the synthetic-only reader using gold (crop, string) pairs taken
ONLY from the other folds, then decode the held-out fold.  A final model trained on all
529 training images decodes the test split.  This keeps the out-of-fold estimate honest
while still letting the reader learn real receipt glyphs.
"""
import os, pickle, json, re, sys, time, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from crnn import CRNN, pad_batch, dev
from crnn2 import encode2, decode2
from synth2 import NCLS2, C2I2
from ctc_select import gold_losses

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
OKPAT = re.compile(r'^[-]?[0-9A-Za-z][0-9A-Za-z.,:%]*$')
BASE = os.environ.get('READER_BASE', 'crnn2.pt')   # synthetic-only: never saw an answer
LOGF = W + r'\progress_reader.txt'


def log(msg):
    with open(LOGF, 'a') as f:
        f.write(str(msg) + chr(10))
        f.flush()
    print(msg, flush=True)


def select_pairs(topk=2, thr=0.35):
    """(image index -> [(crop, gold), ...]) chosen by gold CTC loss."""
    cache = W + r'\cache\sel_pairs.pkl'
    if os.path.exists(cache):
        return pickle.load(open(cache, 'rb'))
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    tok = pickle.load(open(W + r'\cache\tok3_train.pkl', 'rb'))
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + BASE, map_location=dev))
    net.eval()
    out = []
    t0 = time.time()
    for i, ((strips, F), g) in enumerate(zip(tok, golds)):
        L = gold_losses(net, strips, g)
        if len(L) == 0:
            out.append([]); continue
        o = np.argsort(L)[:topk]
        out.append([(strips[int(j)], g) for j in o if L[j] < thr])
        if i % 100 == 0:
            print(f'  select {i} ({time.time()-t0:.0f}s)', flush=True)
    pickle.dump(out, open(cache, 'wb'), protocol=4)
    return out


def pseudo_pool():
    """Confident readings of every strip (train+test) from the uncontaminated model."""
    X, Y = [], []
    for split in ('train', 'test'):
        tok = pickle.load(open(W + f'\\cache\\tok3_{split}.pkl', 'rb'))
        dec = pickle.load(open(W + f'\\cache\\dec4_{split}.pkl', 'rb'))
        for (strips, F), (txt, cf, _) in zip(tok, dec):
            for s, t, c in zip(strips, txt, cf):
                t = t.strip()
                if (not t or len(t) > 22 or c <= -0.05 or len(t) < 2
                        or not OKPAT.match(t) or not all(ch in C2I2 for ch in t)):
                    continue
                X.append(s); Y.append(t)
    return X, Y


def jitter(im, rng):
    import cv2
    a = im.astype(np.float32) / 255.
    if rng.random() < 0.5:
        a = np.clip(a * rng.uniform(0.8, 1.25) + rng.uniform(-0.14, 0.14), 0, 1)
    if rng.random() < 0.35:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.1))
    if rng.random() < 0.35:
        a = np.clip(a + rng.normal(0, rng.uniform(0.01, 0.06), a.shape), 0, 1)
    if rng.random() < 0.35:            # horizontal stretch / squeeze
        w = max(8, int(a.shape[1] * rng.uniform(0.85, 1.18)))
        a = cv2.resize(a, (w, a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (a * 255).astype(np.uint8)


def adapt(pairs, PX, PY, SX, SY, epochs=2, bs=112, lr=7e-4, wgold=14, nsyn=14000, nps=12000):
    rng = np.random.default_rng(0)
    X = [p[0] for p in pairs]; Y = [p[1] for p in pairs]; Wt = [wgold] * len(pairs)
    sel = rng.choice(len(PX), size=min(nps, len(PX)), replace=False)
    X += [PX[i] for i in sel]; Y += [PY[i] for i in sel]; Wt += [1] * len(sel)
    sel = rng.choice(len(SX), size=min(nsyn, len(SX)), replace=False)
    X += [SX[i] for i in sel]; Y += [SY[i] for i in sel]; Wt += [1] * len(sel)
    net = CRNN(ncls=NCLS2).to(dev)
    net.load_state_dict(torch.load(W + '\\cache\\' + BASE, map_location=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler('cuda')
    rep = [i for i, w in enumerate(Wt) for _ in range(w)]
    steps = max(1, len(rep) // bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs * steps, pct_start=0.2)
    jr = np.random.default_rng(1)
    for ep in range(epochs):
        order = sorted(rep, key=lambda i: X[i].shape[1])
        buckets = [order[k * bs:(k + 1) * bs] for k in range(steps)]
        random.shuffle(buckets)
        net.train(); tot = 0.0
        for b in buckets:
            xb, ws = pad_batch([jitter(X[i], jr) for i in b])
            xb = torch.from_numpy(xb).float().div(255.).unsqueeze(1).to(dev)
            tgt, tl = encode2([Y[i] for i in b])
            with torch.amp.autocast('cuda'):
                lg = net(xb)
            lp = lg.float().log_softmax(-1).permute(1, 0, 2)
            il = torch.full((len(b),), lp.shape[0], dtype=torch.long)
            loss = ctc(lp, tgt, il, tl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sch.step()
            tot += loss.item()
        log(f'    ep{ep} loss {tot/steps:.4f}')
    del opt, scaler, X, Y, Wt, rep
    torch.cuda.empty_cache()
    return net


SCALES = (1.0, 0.88, 1.14)


def _rescale(im, f):
    if f == 1.0:
        return im
    import cv2
    w = max(8, int(round(im.shape[1] * f)))
    return cv2.resize(im, (w, im.shape[0]), interpolation=cv2.INTER_LINEAR)


def decode_images(net, tok, idxs, bs=96):
    """Width-scale TTA: each strip is read at several horizontal scales and every
    reading is kept, because candidate recall is a union over readings."""
    flat = []
    for i in idxs:
        flat.extend(tok[i][0])
    if not flat:
        return {i: ([], np.zeros(0, np.float32)) for i in idxs}
    per_scale = []
    net.eval()
    for f in SCALES:
        sc = [_rescale(s, f) for s in flat]
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
        del sc
        torch.cuda.empty_cache()
    res, p = {}, 0
    for i in idxs:
        n = len(tok[i][0])
        t = [x for ts, _ in per_scale for x in ts[p:p + n]]
        c = np.concatenate([cs[p:p + n] for _, cs in per_scale])
        res[i] = (t, c); p += n
    return res


def main():
    tr = pd.read_csv(D + r'\train.csv'); te = pd.read_csv(D + r'\test.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    ntr, nte = len(tr), len(te)
    print('selecting real pairs...', flush=True)
    pairs = select_pairs()
    print('pairs on', sum(1 for p in pairs if p), 'images', flush=True)
    PX, PY = pseudo_pool()
    log(f'pseudo pool {len(PX)}')
    SX, SY = pickle.load(open(W + r'\cache\synth2.pkl', 'rb'))
    tokT = pickle.load(open(W + r'\cache\tok3_train.pkl', 'rb'))
    tokE = pickle.load(open(W + r'\cache\tok3_test.pkl', 'rb'))
    NI = np.array([min(a['num_items'], 8) for a in A])

    out_tr = [None] * ntr
    for k, (ti, vi) in enumerate(StratifiedKFold(3, shuffle=True, random_state=0)
                                 .split(np.zeros(ntr), NI)):
        log(f'fold {k}')
        P = [q for i in ti for q in pairs[i]]
        net = adapt(P, PX, PY, SX, SY)
        torch.save(net.state_dict(), os.path.join(W, 'cache', f'crnn_fold{k}.pt'))
        r = decode_images(net, tokT, list(vi))
        for i in vi:
            out_tr[i] = (r[i][0], r[i][1], np.tile(tokT[i][1], (len(SCALES), 1)))
        hit = np.mean([golds[i] in set(x.strip() for x in r[i][0]) for i in vi])
        log(f'  fold {k} exact-decode recall {hit:.4f}')
        del net; torch.cuda.empty_cache()
    pickle.dump(out_tr, open(W + r'\cache\dec3_train.pkl', 'wb'), protocol=4)
    rec = np.mean([golds[i] in set(x.strip() for x in out_tr[i][0]) for i in range(ntr)])
    log(f'OOF exact-decode recall {float(rec):.4f}')

    log('full model for test')
    P = [q for p in pairs for q in p]
    net = adapt(P, PX, PY, SX, SY)
    torch.save(net.state_dict(), W + r'\cache\crnn_full.pt')
    r = decode_images(net, tokE, list(range(nte)))
    out_te = [(r[i][0], r[i][1], np.tile(tokE[i][1], (len(SCALES), 1))) for i in range(nte)]
    pickle.dump(out_te, open(W + r'\cache\dec3_test.pkl', 'wb'), protocol=4)
    log('done')


if __name__ == '__main__':
    main()
