"""Wider, more receipt-like synthetic corpus.

Two changes over synth2: a much larger font pool (every TrueType face installed locally,
falling back to the fonts bundled with matplotlib), and a degradation chain that models
thermal-printer artefacts -- dot-matrix dropout, ink bleed, exposure banding, heavier
condensation -- on top of the resolution bottleneck.
"""
import os, glob, string, sys
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib

from synth2 import ALPHA2, C2I2, WORDS, SUFFIX
from synth import PATTERNS, PW, SH


def font_pool():
    d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
    fonts = [os.path.join(d, f) for f in
             ['DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf', 'DejaVuSans.ttf',
              'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'cmtt10.ttf', 'cmss10.ttf',
              'DejaVuSansMono-Oblique.ttf', 'DejaVuSans-Oblique.ttf']
             if os.path.exists(os.path.join(d, f))]
    sysdirs = [r'C:\Windows\Fonts', '/usr/share/fonts', '/usr/local/share/fonts',
               os.path.expanduser('~/.fonts')]
    extra = []
    for sd in sysdirs:
        if os.path.isdir(sd):
            extra += glob.glob(os.path.join(sd, '**', '*.ttf'), recursive=True)
    bad = ('wingding', 'webding', 'symbol', 'marlett', 'bookshelf', 'holomdl',
           'segmdl', 'segoeicons', 'mtextra', 'outlook', 'opensymbol')
    extra = [f for f in extra if not any(b in os.path.basename(f).lower() for b in bad)]
    ok = []
    for f in extra:
        try:
            fo = ImageFont.truetype(f, 32)
            if fo.getbbox('0123456789')[2] <= 20 or fo.getbbox('A')[2] <= 3:
                continue
            # reject faces whose digits all render to the same glyph (missing -> .notdef box)
            sigs = set()
            for c in '0123456789':
                im = Image.new('L', (48, 48), 255)
                ImageDraw.Draw(im).text((6, 4), c, font=fo, fill=0)
                a = np.array(im) < 128
                if a.sum() < 4:
                    sigs.add(('blank', c)); continue
                ys, xs = np.where(a)
                g = cv2.resize(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8),
                               (8, 8), interpolation=cv2.INTER_AREA)
                sigs.add((g > 0).tobytes())
            if len(sigs) >= 9:
                ok.append(f)
        except Exception:
            pass
    return fonts + ok


FONTS = font_pool()


def rand_text(rng):
    r = rng.random()
    if r < 0.52:
        p = PATTERNS[rng.choice(len(PATTERNS), p=PW)]
        sep = rng.choice([',', '.']); other = '.' if sep == ',' else ','
        t = ''.join(str(rng.integers(0, 10)) if c == 'D' else c
                    for c in p.format(s=sep, t=other))
        if rng.random() < 0.10:
            t = '-' + t
        if rng.random() < 0.05:
            t += '%'
        if rng.random() < 0.04:
            t = 'Rp' + t
        return t
    if r < 0.90:
        w = (WORDS[rng.integers(0, len(WORDS))] if rng.random() < 0.55 else
             ''.join(rng.choice(list(string.ascii_uppercase))
                     for _ in range(int(rng.integers(2, 13)))))
        if rng.random() < 0.25:
            w = w + SUFFIX[rng.integers(0, len(SUFFIX))]
        if rng.random() < 0.12:
            w = w.lower()
        return w.strip() or 'TOTAL'
    # label + value on one strip (whole-line crops look like this)
    w = WORDS[rng.integers(0, len(WORDS))]
    p = PATTERNS[rng.choice(len(PATTERNS), p=PW)]
    sep = rng.choice([',', '.'])
    v = ''.join(str(rng.integers(0, 10)) if c == 'D' else c
                for c in p.format(s=sep, t='.' if sep == ',' else ','))
    return w + ' ' * int(rng.integers(1, 4)) + v


def render(txt, rng):
    fp = FONTS[rng.integers(0, len(FONTS))]
    size = int(rng.integers(26, 60))
    try:
        font = ImageFont.truetype(fp, size)
    except Exception:
        font = ImageFont.load_default()
    pad = int(size * 0.4)
    ls = int(rng.integers(-1, max(2, int(size * 0.14))))
    widths = []
    for c in txt:
        try:
            bb = font.getbbox(c); w = bb[2] - bb[0]
        except Exception:
            w = size // 2
        widths.append(max(w, size // 5) if c != ' ' else max(size // 3, 4))
    Wd = sum(widths) + ls * max(0, len(txt) - 1) + 2 * pad
    Ht = int(size * 2.2)
    if Wd > 4000 or Ht > 400:
        return None
    img = Image.new('L', (Wd, Ht), 255)
    d = ImageDraw.Draw(img)
    x = pad; y = int(Ht * 0.5 - size * 0.78)
    for c, w in zip(txt, widths):
        if c != ' ':
            d.text((x, y), c, font=font, fill=0)
        x += w + ls
    a = np.array(img, np.float32) / 255.
    ink = a < 0.6
    if ink.sum() < 3:
        return None
    ys, xs = np.where(ink)
    lh = ys.max() - ys.min() + 1
    m = max(1, int(lh * rng.uniform(0.05, 0.30)))
    my = max(1, int(lh * rng.uniform(0.08, 0.50)))
    return a[max(0, ys.min() - my):min(Ht, ys.max() + 1 + my),
             max(0, xs.min() - m):min(Wd, xs.max() + 1 + m)]


def degrade(a, rng):
    h, w = a.shape
    k = max(1, int(round(h * 0.045)))
    r = rng.random()
    if r < 0.30:
        a = cv2.erode(a, np.ones((k, k), np.uint8))       # heavier ink
    elif r < 0.52:
        a = cv2.dilate(a, np.ones((k, k), np.uint8))      # starved ink
    if rng.random() < 0.25:                               # ink bleed
        a = cv2.GaussianBlur(cv2.erode(a, np.ones((k, k), np.uint8)), (0, 0),
                             rng.uniform(0.5, 1.4))
    if rng.random() < 0.22:                               # dot-matrix dropout
        m = (rng.random((max(2, h // 2), max(2, w // 2))) > rng.uniform(0.05, 0.22)).astype(np.float32)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        a = np.maximum(a, 1.0 - m)
    ang = rng.uniform(-4.0, 4.0); shear = rng.uniform(-0.18, 0.18)
    sx = rng.uniform(0.62, 1.35); sy = rng.uniform(0.9, 1.15)
    M = np.float32([[sx * np.cos(np.deg2rad(ang)), shear, 2],
                    [np.sin(np.deg2rad(ang)) * 0.5, sy, 2]])
    nw = int(w * sx * 1.05 + abs(shear) * h) + 4
    nh = int(h * sy) + 5
    a = cv2.warpAffine(a, M, (max(nw, 6), max(nh, 6)), borderValue=1.0, flags=cv2.INTER_LINEAR)
    hh, ww = a.shape
    if rng.random() < 0.88:                               # resolution bottleneck
        hs = int(rng.integers(7, 30))
        a = cv2.resize(a, (max(4, int(round(ww * hs / hh))), hs), interpolation=cv2.INTER_AREA)
        hh, ww = a.shape
    sc = SH / hh
    a = cv2.resize(a, (max(8, int(ww * sc)), SH),
                   interpolation=cv2.INTER_LINEAR if sc > 1 else cv2.INTER_AREA)
    if rng.random() < 0.82:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.2, 1.7))
    if rng.random() < 0.15:
        kk = int(rng.integers(2, 6)); ker = np.zeros((kk, kk), np.float32); ker[kk // 2, :] = 1.0 / kk
        a = cv2.filter2D(a, -1, ker)
    lo = rng.uniform(0.0, 0.50); hi = rng.uniform(0.68, 1.0)
    a = lo + a * (hi - lo)
    if rng.random() < 0.30:                               # exposure banding
        nb = int(rng.integers(2, 6))
        band = rng.uniform(-0.12, 0.12, nb)
        a = a + cv2.resize(band.astype(np.float32).reshape(-1, 1),
                           (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    a = a + np.linspace(rng.uniform(-.12, .12), rng.uniform(-.12, .12), a.shape[0])[:, None]
    a = a + np.linspace(rng.uniform(-.14, .14), rng.uniform(-.14, .14), a.shape[1])[None, :]
    a = a + rng.normal(0, rng.uniform(0.005, 0.075), a.shape)
    if rng.random() < 0.08:
        m = rng.random(a.shape) < rng.uniform(0.002, 0.02)
        a[m] = rng.random(int(m.sum()))
    if rng.random() < 0.28:
        q = int(rng.integers(20, 80))
        enc = cv2.imencode('.jpg', (np.clip(a, 0, 1) * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        a = cv2.imdecode(enc, 0).astype(np.float32) / 255.
    return np.clip(a, 0, 1)


def gen_batch(n, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    guard = 0
    while len(X) < n and guard < n * 40:
        guard += 1
        t = ''.join(c for c in rand_text(rng) if c in C2I2 or c == ' ')
        if not t.strip():
            continue
        a = render(t, rng)
        if a is None:
            continue
        a = degrade(a, rng)
        if a.shape[1] < 8 or a.shape[1] > 420:
            continue
        X.append((a * 255).astype(np.uint8)); Y.append(t)
    return X, Y


if __name__ == '__main__':
    import pickle, time
    from concurrent.futures import ProcessPoolExecutor
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 300000
    print('fonts available:', len(FONTS), flush=True)
    t0 = time.time(); ch = 16
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(gen_batch, [N // ch] * ch, list(range(500, 500 + ch))))
    X = [a for r in res for a in r[0]]; Y = [y for r in res for y in r[1]]
    print(len(X), 'samples', round(time.time() - t0, 1), 's', flush=True)
    pickle.dump((X, Y), open(r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work\cache\synth3.pkl',
                             'wb'), protocol=4)
