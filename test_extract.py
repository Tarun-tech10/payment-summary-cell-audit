import os, json, re, sys, time
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from ocr_extract import extract, D

ALPHA = '0123456789,.'

def pattern(s):
    return ''.join('D' if c.isdigit() else 'S' for c in s if c in ALPHA)

def tok_pattern(tk, lh):
    bx = tk['boxes']
    hs = bx[:, 3].astype(float)
    big = hs[hs >= lh * 0.5]
    ref = np.median(big) if len(big) else np.median(hs)
    return ''.join('D' if h >= ref * 0.62 else 'S' for h in hs)

def work(args):
    i, path, total = args
    try:
        E = extract(path)
    except Exception as e:
        return i, 0, 0, str(e)
    s = ''.join(c for c in total if c in ALPHA)
    pat = pattern(total)
    nmatch = 0; ntok = 0
    for L in E['lines']:
        for tk in L['tokens']:
            ntok += 1
            if len(tk['boxes']) != len(pat):
                continue
            if tok_pattern(tk, L['lh']) == pat:
                nmatch += 1
    return i, nmatch, ntok, ''

if __name__ == '__main__':
    t = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in t.answer_json]
    jobs = [(i, os.path.join(D, t.image_path[i].replace('/', os.sep)), A[i]['total_price'])
            for i in range(len(t))]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(jobs)
    jobs = jobs[:n]
    t0 = time.time()
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(work, jobs, chunksize=4))
    nm = np.array([r[1] for r in res]); nt = np.array([r[2] for r in res])
    print('images', len(res), 'time', round(time.time()-t0, 1))
    print('has >=1 pattern match:', (nm > 0).mean())
    print('mean matches per img:', nm.mean(), 'mean tokens', nt.mean())
    print('errors', [r[3] for r in res if r[3]][:3])
    print('match-count hist', np.bincount(np.minimum(nm, 10)))
