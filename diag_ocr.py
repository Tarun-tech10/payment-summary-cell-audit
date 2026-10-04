import pickle, json, re, os
import numpy as np, pandas as pd, cv2
D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
OUT = r'C:\Users\tarun\AppData\Local\Temp\claude\c--Users-tarun-OneDrive-Desktop-ERIS\21c157a1-9ebf-43b4-a7b6-8a9e02cd49da\scratchpad'
t = pd.read_csv(D + r'\train.csv'); A = [json.loads(x) for x in t.answer_json]
dec = pickle.load(open(r'cache\dec_train.pkl', 'rb'))
tok = pickle.load(open(r'cache\tok_train.pkl', 'rb'))

def dg(s): return re.sub(r'\D', '', s)

def ed(a, b):
    m, n = len(a), len(b)
    if not m or not n: return max(m, n)
    p = list(range(n + 1))
    for i in range(m):
        c = [i + 1] + [0] * n
        for j in range(n):
            c[j + 1] = min(p[j] + (a[i] != b[j]), p[j + 1] + 1, c[j] + 1)
        p = c
    return p[n]

rows = []
best_imgs = []
for i, a in enumerate(A):
    texts, confs, F = dec[i]
    g = a['total_price'].strip(); gd = dg(g)
    if len(texts) == 0:
        rows.append((i, g, '', 9, 0)); continue
    ds = [dg(x) for x in texts]
    d = [ed(x, gd) for x in ds]
    j = int(np.argmin(d))
    rows.append((i, g, texts[j], d[j], F[j, 16]))
R = pd.DataFrame(rows, columns=['i', 'gold', 'best', 'dist', 'p3'])
print('images where a token has digit-dist 0:', (R.dist == 0).mean())
print('dist<=1:', (R.dist <= 1).mean(), 'dist<=2:', (R.dist <= 2).mean())
print(R.dist.value_counts().head(8))
print(R[R.dist > 0].head(25).to_string())
# strip montage for failures
bad = R[R.dist >= 2].i.values[:14]
tiles = []
for i in bad:
    strips, F = tok[i]
    texts = dec[i][0]
    gd = dg(A[i]['total_price'])
    order = np.argsort([ed(dg(x), gd) for x in texts])[:3]
    for j in order:
        s = strips[j]
        s = np.pad(s, ((2, 2), (2, 200 - min(s.shape[1], 190))), constant_values=255)[:, :200]
        cv2.putText(s, f"{A[i]['total_price']}|{texts[j]}", (2, 12), 0, 0.35, 0, 1)
        tiles.append(s)
cv2.imwrite(os.path.join(OUT, 'ocrfail.png'), np.vstack(tiles))
print('wrote ocrfail.png', len(tiles))
