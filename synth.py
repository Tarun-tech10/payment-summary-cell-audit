"""Synthetic receipt-number strip renderer (matplotlib's bundled DejaVu fonts)."""
import os, random, glob
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib

FDIR = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
FONTS = [os.path.join(FDIR, f) for f in
         ['DejaVuSansMono.ttf', 'DejaVuSansMono-Bold.ttf', 'DejaVuSans.ttf',
          'DejaVuSans-Bold.ttf', 'DejaVuSerif.ttf', 'cmtt10.ttf', 'cmss10.ttf',
          'DejaVuSansMono-Oblique.ttf', 'DejaVuSans-Oblique.ttf']]
FONTS = [f for f in FONTS if os.path.exists(f)]
ALPHA = '0123456789,.-'
C2I = {c: i + 1 for i, c in enumerate(ALPHA)}   # 0 = CTC blank
NCLS = len(ALPHA) + 1
SH = 32  # strip height

PATTERNS = ['DD{s}DDD', 'DDD{s}DDD', 'D{s}DDD', 'DDDDD', 'DDDD', 'DDDDDD',
            'D{s}DDD{s}DDD', 'DDD{s}DDD{t}DD', 'DD{s}DDD{t}DD', 'DD{s}DDD',
            'DD{s}DDD', 'DDD', 'DD', 'D', 'DD{s}DDD', 'DDDDDDD', 'DD{s}DDD{t}DD']
PW = np.array([0.24, 0.14, 0.05, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.08,
               0.08, 0.03, 0.03, 0.02, 0.08, 0.02, 0.03])
PW = PW / PW.sum()


def rand_text(rng):
    p = PATTERNS[rng.choice(len(PATTERNS), p=PW)]
    sep = rng.choice([',', '.'])
    other = '.' if sep == ',' else ','
    s = p.format(s=sep, t=other)
    out = []
    for ch in s:
        out.append(str(rng.integers(0, 10)) if ch == 'D' else ch)
    txt = ''.join(out)
    if rng.random() < 0.10:
        txt = '-' + txt
    if rng.random() < 0.06:
        txt = txt + rng.choice(['%', ''])
        txt = txt.replace('%', '')
    return txt


def render(txt, rng):
    fp = FONTS[rng.integers(0, len(FONTS))]
    size = int(rng.integers(28, 56))
    try:
        font = ImageFont.truetype(fp, size)
    except Exception:
        font = ImageFont.load_default()
    pad = int(size * 0.4)
    # letter spacing
    ls = int(rng.integers(-1, max(2, int(size * 0.12))))
    widths = []
    for c in txt:
        try:
            w = font.getbbox(c)[2] - font.getbbox(c)[0]
        except Exception:
            w = size // 2
        widths.append(max(w, size // 4))
    W = sum(widths) + ls * max(0, len(txt) - 1) + 2 * pad
    H = int(size * 2.0)
    img = Image.new('L', (W, H), 255)
    d = ImageDraw.Draw(img)
    x = pad
    y = int(H * 0.5 - size * 0.72)
    for c, w in zip(txt, widths):
        d.text((x, y), c, font=font, fill=0)
        x += w + ls
    a = np.array(img, np.float32) / 255.
    # crop to ink bbox with padding similar to real extraction
    ink = a < 0.6
    if ink.sum() < 3:
        return None
    ys, xs = np.where(ink)
    lh = ys.max() - ys.min() + 1
    m = max(1, int(lh * rng.uniform(0.06, 0.25)))
    my = max(1, int(lh * rng.uniform(0.10, 0.45)))
    y0 = max(0, ys.min() - my); y1 = min(H, ys.max() + 1 + my)
    x0 = max(0, xs.min() - m); x1 = min(W, xs.max() + 1 + m)
    a = a[y0:y1, x0:x1]
    return a


def degrade(a, rng):
    h, w = a.shape
    # thickness
    r = rng.random()
    k = max(1, int(round(h * 0.04)))
    if r < 0.3:
        a = cv2.erode(a, np.ones((k, k), np.uint8))      # bolder (dark=low)
    elif r < 0.5:
        a = cv2.dilate(a, np.ones((k, k), np.uint8))     # thinner
    # geometric: shear / rotate / squeeze
    ang = rng.uniform(-3.5, 3.5)
    shear = rng.uniform(-0.15, 0.15)
    sx = rng.uniform(0.78, 1.25)
    M = np.float32([[sx * np.cos(np.deg2rad(ang)), shear, 0],
                    [np.sin(np.deg2rad(ang)) * 0.5, 1, 0]])
    nw = int(w * sx * 1.05 + abs(shear) * h) + 2
    M[0, 2] += 2; M[1, 2] += 2
    a = cv2.warpAffine(a, M, (nw, h + 4), borderValue=1.0, flags=cv2.INTER_LINEAR)
    # resolution bottleneck: real crops come from 9-26 px tall text lines
    hh, ww = a.shape
    if rng.random() < 0.85:
        hs = int(rng.integers(8, 27))
        ws_ = max(4, int(round(ww * hs / hh)))
        a = cv2.resize(a, (ws_, hs), interpolation=cv2.INTER_AREA)
        if rng.random() < 0.3:
            a = a + rng.normal(0, rng.uniform(0.01, 0.05), a.shape).astype(np.float32)
        hh, ww = a.shape
    scale = SH / hh
    a = cv2.resize(a, (max(8, int(ww * scale)), SH),
                   interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    # blur
    if rng.random() < 0.8:
        s = rng.uniform(0.2, 1.6)
        a = cv2.GaussianBlur(a, (0, 0), s)
    if rng.random() < 0.15:
        kk = int(rng.integers(2, 5))
        ker = np.zeros((kk, kk), np.float32); ker[kk // 2, :] = 1.0 / kk
        a = cv2.filter2D(a, -1, ker)
    # contrast / brightness / background gradient
    lo = rng.uniform(0.0, 0.45); hi = rng.uniform(0.72, 1.0)
    a = lo + a * (hi - lo)
    gy = np.linspace(rng.uniform(-.12, .12), rng.uniform(-.12, .12), a.shape[0])[:, None]
    gx = np.linspace(rng.uniform(-.12, .12), rng.uniform(-.12, .12), a.shape[1])[None, :]
    a = a + gy + gx
    a = a + rng.normal(0, rng.uniform(0.005, 0.06), a.shape)
    if rng.random() < 0.25:  # jpeg-ish
        q = int(rng.integers(25, 80))
        enc = cv2.imencode('.jpg', (np.clip(a, 0, 1) * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, q])[1]
        a = cv2.imdecode(enc, 0).astype(np.float32) / 255.
    return np.clip(a, 0, 1)


def gen_batch(n, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        txt = rand_text(rng)
        a = render(txt, rng)
        if a is None:
            continue
        a = degrade(a, rng)
        if a.shape[1] < 8 or a.shape[1] > 400:
            continue
        X.append((a * 255).astype(np.uint8))
        Y.append(txt)
    return X, Y


if __name__ == '__main__':
    import sys, pickle, time
    from concurrent.futures import ProcessPoolExecutor
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
    t0 = time.time()
    chunks = 16
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(gen_batch, [N // chunks] * chunks, list(range(chunks))))
    X = [a for r in res for a in r[0]]
    Y = [y for r in res for y in r[1]]
    print(len(X), 'samples', round(time.time() - t0, 1), 's')
    with open(r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work\cache\synth.pkl', 'wb') as f:
        pickle.dump((X, Y), f, protocol=4)
