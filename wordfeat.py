"""Keyword-presence features derived from decoded receipt tokens."""
import re
import numpy as np

F_RELY, F_RELX, F_X1, F_TW, F_LHR, F_NG, F_HM, F_HS, F_LI, F_LIB, F_TI, F_NTOK, \
    F_RIGHT, F_P0, F_P1, F_P2, F_P3, F_YC, F_XC, F_LH, F_NLINES, F_RH, F_RW, F_SW = range(24)

KEYS = {
    'total':    ['TOTAL', 'TOTL', 'T0TAL', 'TOTA', 'OTAL'],
    'subtotal': ['SUBTOTAL', 'SUBTOTA', 'UBTOTAL', 'SUB'],
    'grand':    ['GRAND', 'GRND'],
    'cash':     ['CASH', 'TUNAI', 'CASHIER'],
    'change':   ['CHANGE', 'KEMBALI', 'KEMBALIAN'],
    'tax':      ['PPN', 'PAJAK', 'TAX', 'PB1', 'PPH', 'VAT'],
    'service':  ['SERVICE', 'SERV', 'SVC', 'LAYANAN', 'CHARGE'],
    'disc':     ['DISKON', 'DISC', 'DISCOUNT', 'POTONGAN'],
    'item':     ['ITEM', 'ITEMS', 'QTY', 'JUMLAH'],
    'bayar':    ['BAYAR', 'PEMBAYARAN', 'TAGIHAN', 'DUE', 'AMOUNT'],
    'round':    ['PEMBULATAN', 'ROUNDING', 'ROUND'],
    'plastik':  ['PLASTIK', 'KANTONG', 'KRESEK'],
    'nett':     ['NETT', 'NET', 'GROSS', 'DPP'],
}


def norm(t):
    return re.sub(r'[^A-Z]', '', t.upper())


def sim(a, b):
    """cheap normalised similarity: longest-common-subsequence ratio."""
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = [0] * (lb + 1)
    for i in range(la):
        cur = [0] * (lb + 1)
        for j in range(lb):
            cur[j + 1] = prev[j] + 1 if a[i] == b[j] else max(prev[j + 1], cur[j])
        prev = cur
    return prev[lb] / max(la, lb)


def token_keyscores(texts):
    """(n_tokens, n_keys) best fuzzy similarity to each keyword group."""
    ks = list(KEYS)
    out = np.zeros((len(texts), len(ks)), np.float32)
    cache = {}
    for i, t in enumerate(texts):
        w = norm(t)
        if len(w) < 2:
            continue
        if w in cache:
            out[i] = cache[w]; continue
        v = np.array([max(sim(w, k) for k in KEYS[g]) for g in ks], np.float32)
        cache[w] = v; out[i] = v
    return out, ks


def image_wordfeats(dec_row):
    texts, confs, F = dec_row
    ks = list(KEYS)
    if len(texts) == 0:
        return np.zeros(len(ks) * 5 + 4, np.float32)
    S, _ = token_keyscores(texts)
    rely = F[:, F_RELY]
    p3 = F[:, F_P3]; p2 = F[:, F_P2]
    out = []
    for j in range(len(ks)):
        s = S[:, j]
        out += [float(s.max()), float((s > 0.85).sum()), float((s > 0.7).sum()),
                float(rely[s.argmax()]), float((s > 0.8).astype(np.float32) @ (p2 + p3) )]
    out += [float(len(texts)), float(F[0, F_NLINES]),
            float((S.max(1) > 0.8).mean()), float(np.mean(confs))]
    return np.array(out, np.float32)
