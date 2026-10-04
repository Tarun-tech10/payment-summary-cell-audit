import json, numpy as np, pandas as pd
from collections import Counter
from scorer import breakdown, grid_score, fields_score, total_price_score
D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
t = pd.read_csv(D+r'\train.csv')
A = [json.loads(x) for x in t.answer_json]

# best constant grid: search among observed grids + greedy cellwise
best = None
cands = [tuple(a['semantic_grid']) for a in A]
for g in set(cands):
    s = np.mean([grid_score(list(g), a['semantic_grid']) for a in A])
    if best is None or s > best[0]:
        best = (s, g)
print("best constant grid", best[0], np.array(best[1]).reshape(4,4).tolist())

# greedy cellwise improvement
g = list(best[1])
improved = True
while improved:
    improved = False
    for i in range(16):
        cur = g[i]
        for v in [0,1,2,3]:
            if v == cur: continue
            g2 = list(g); g2[i] = v
            s = np.mean([grid_score(g2, a['semantic_grid']) for a in A])
            if s > best[0] + 1e-9:
                best = (s, tuple(g2)); g = g2; improved = True
print("greedy grid", best[0], np.array(g).reshape(4,4).tolist())

# fields
fbest = None
for combo in [dict(zip(['subtotal_present','service_present','tax_present','etc_present'], c))
              for c in __import__('itertools').product([False,True],repeat=4)]:
    s = np.mean([fields_score(combo, a['fields_present']) for a in A])
    if fbest is None or s > fbest[0]: fbest = (s, combo)
print("best fields", fbest)

# num_items
nb = max(set(a['num_items'] for a in A), key=lambda v: sum(1 for a in A if a['num_items']==v))
print("best num_items", nb, np.mean([a['num_items']==nb for a in A]))

# total
tb = None
for c in set(a['total_price'] for a in A):
    s = np.mean([total_price_score(c, a['total_price']) for a in A])
    if tb is None or s > tb[0]: tb = (s, c)
print("best const total", tb)
print("TOTAL const score", 0.45*best[0]+0.35*tb[0]+0.12*np.mean([a['num_items']==nb for a in A])+0.08*fbest[0])
