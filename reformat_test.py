"""Do canonical re-formats of read digits recover the exactly-printed total?"""
import json, pickle, re
from collections import Counter
import numpy as np, pandas as pd

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
W = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work'
NUM = re.compile(r'-?\d[\d.,]*\d|\d')


def dg(s):
    return re.sub(r'\D', '', s)


def group3(d, sep):
    if len(d) <= 3:
        return d
    out = []
    while len(d) > 3:
        out.append(d[-3:]); d = d[:-3]
    out.append(d)
    return sep.join(reversed(out))


def variants(d, sep_style):
    """Plausible printed forms of a digit string."""
    v = {d}
    for s in (',', '.'):
        v.add(group3(d, s))
    if len(d) > 2:
        for s in (',', '.'):
            head, tail = d[:-2], d[-2:]
            if head:
                v.add(group3(head, ',') + s + tail)
                v.add(group3(head, '.') + s + tail)
                v.add(head + s + tail)
    if sep_style:
        v.add(group3(d, sep_style))
    return {x for x in v if x}


def receipt_sep(texts):
    c = Counter()
    for t in texts:
        for m in NUM.findall(t):
            if len(dg(m)) >= 4:
                for ch in m:
                    if ch in ',.':
                        c[ch] += 1
    return c.most_common(1)[0][0] if c else ''


if __name__ == '__main__':
    tr = pd.read_csv(D + r'\train.csv')
    A = [json.loads(x) for x in tr.answer_json]
    golds = [a['total_price'].strip() for a in A]
    n = len(A)
    for name in ['dec2', 'dec2B']:
        dec = pickle.load(open(W + f'\\cache\\{name}_train.pkl', 'rb'))
        raw = ref = both = digok = 0
        sepacc = 0
        for i in range(n):
            texts = dec[i][0]
            cands = set()
            digits = set()
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
            gsep = ''.join(ch for ch in golds[i] if ch in ',.')
            if gsep and sep == gsep[0]:
                sepacc += 1
            rf = set()
            for d in digits:
                if 2 <= len(d) <= 9:
                    rf |= variants(d, sep)
            raw += golds[i] in cands
            ref += golds[i] in rf
            both += golds[i] in (cands | rf)
            digok += dg(golds[i]) in digits
        print(f'{name}: raw {raw/n:.4f}  reformat-only {ref/n:.4f}  union {both/n:.4f}'
              f'  digits-present {digok/n:.4f}  sep-style acc {sepacc/n:.4f}')
