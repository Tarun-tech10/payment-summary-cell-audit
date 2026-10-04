"""Compare candidate base readers on the real strip cache (digit recall is what counts)."""
import os, sys, json, pickle, time
import numpy as np, pandas as pd, torch
from crnn import CRNN, pad_batch, dev
from synth2 import NCLS2
from eval_reader import decode_all, report

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
LOGF = W + r'\progress_cmp.txt'


def log(m):
    with open(LOGF, 'a') as f:
        f.write(str(m) + chr(10)); f.flush()
    print(m, flush=True)


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv')
    golds = [json.loads(x)['total_price'].strip() for x in tr.answer_json]
    tok = pickle.load(open(W + r'\cache\tok3_train.pkl', 'rb'))
    for ck in sys.argv[1:] or ['crnn2.pt', 'crnn3.pt']:
        p = W + '\\cache\\' + ck
        if not os.path.exists(p):
            log(f'{ck}: missing'); continue
        net = CRNN(ncls=NCLS2).to(dev)
        net.load_state_dict(torch.load(p, map_location=dev))
        t0 = time.time()
        dec = decode_all(net, tok)
        r, f, d = report(dec, golds)
        log(f'{ck} on tok3: raw {r:.4f}  +reformat {f:.4f}  digits {d:.4f}  ({time.time()-t0:.0f}s)')
        del net
        torch.cuda.empty_cache()
