"""Synthetic receipt token renderer: numbers AND words (extended alphabet)."""
import os, string
import numpy as np, cv2
from synth import FONTS, render, degrade, PATTERNS, PW, SH

ALPHA2 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz,.-:%/()@#*+=&'"
C2I2 = {c: i + 1 for i, c in enumerate(ALPHA2)}
I2C2 = {i + 1: c for i, c in enumerate(ALPHA2)}
NCLS2 = len(ALPHA2) + 1

WORDS = """TOTAL SUBTOTAL SUB TOTAL GRAND BAYAR TUNAI KEMBALI KEMBALIAN CASH CHANGE
PPN PB1 PAJAK TAX SERVICE CHARGE SVC DISKON DISC DISCOUNT ITEM ITEMS QTY JUMLAH
HARGA DEBIT CREDIT KARTU TAGIHAN DPP NETT GROSS AMOUNT DUE BALANCE TENDER VISA
MASTER OVO GOPAY QRIS DANA SHOPEEPAY LINKAJA TAKE AWAY DINE IN MEJA TABLE KASIR
PELANGGAN TERIMA KASIH PLASTIK PEMBULATAN ROUNDING BIAYA VOUCHER PROMO MENU
TOTALITEM TOTALQTY PEMBAYARAN PENJUALAN STRUK NOTA FAKTUR RECEIPT ORDER NO
TGL TANGGAL JAM WAKTU CUSTOMER GUEST PAX TYPE CARD MEMBER POIN POINT SALDO
CETAK TOTALDISKON NONTAXABLE TAXABLE INCLUDE INCL EXCL ROUND UP TOTALHARGA
BON KWITANSI ANTAR PESAN LAYANAN ADMIN TUNAI2 EDC BCA BNI BRI MANDIRI GOFOOD
GRABFOOD SUBTOTAL: TOTAL: CASH: PPN10 PPN11 TOTALBAYAR TOTALTAGIHAN""".split()

SUFFIX = ['', ':', '.', '..', '...', ' ', '%', ',']


def rand_word(rng):
    r = rng.random()
    if r < 0.55:
        w = WORDS[rng.integers(0, len(WORDS))]
    elif r < 0.85:
        n = int(rng.integers(2, 13))
        w = ''.join(rng.choice(list(string.ascii_uppercase)) for _ in range(n))
    else:
        n = int(rng.integers(2, 11))
        w = ''.join(rng.choice(list(string.ascii_lowercase)) for _ in range(n))
        if rng.random() < 0.5:
            w = w.capitalize()
    if rng.random() < 0.25:
        w = w + SUFFIX[rng.integers(0, len(SUFFIX))]
    if rng.random() < 0.10:
        w = w.lower()
    return w.strip() or 'TOTAL'


def rand_num(rng):
    p = PATTERNS[rng.choice(len(PATTERNS), p=PW)]
    sep = rng.choice([',', '.'])
    other = '.' if sep == ',' else ','
    s = p.format(s=sep, t=other)
    out = [str(rng.integers(0, 10)) if ch == 'D' else ch for ch in s]
    txt = ''.join(out)
    if rng.random() < 0.10:
        txt = '-' + txt
    if rng.random() < 0.05:
        txt = txt + '%'
    if rng.random() < 0.04:
        txt = 'Rp' + txt
    return txt


def rand_text2(rng):
    r = rng.random()
    if r < 0.50:
        return rand_num(rng)
    if r < 0.90:
        return rand_word(rng)
    return rand_word(rng) + ' ' + rand_num(rng)


def gen_batch2(n, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        txt = rand_text2(rng)
        txt = ''.join(c for c in txt if c in C2I2 or c == ' ')
        if not txt.strip():
            continue
        a = render(txt, rng)
        if a is None:
            continue
        a = degrade(a, rng)
        if a.shape[1] < 8 or a.shape[1] > 420:
            continue
        X.append((a * 255).astype(np.uint8)); Y.append(txt)
    return X, Y


if __name__ == '__main__':
    import sys, pickle, time
    from concurrent.futures import ProcessPoolExecutor
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    t0 = time.time(); chunks = 16
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(gen_batch2, [N // chunks] * chunks, list(range(100, 100 + chunks))))
    X = [a for r in res for a in r[0]]; Y = [y for r in res for y in r[1]]
    print(len(X), 'samples', round(time.time() - t0, 1), 's')
    with open(r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work\cache\synth2.pkl', 'wb') as f:
        pickle.dump((X, Y), f, protocol=4)
