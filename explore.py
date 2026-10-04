import pandas as pd, json, numpy as np
from collections import Counter
D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
t = pd.read_csv(D+r'\train.csv')
te = pd.read_csv(D+r'\test.csv')
print("train", t.shape, "test", te.shape)
A = [json.loads(x) for x in t.answer_json]

ni = [a['num_items'] for a in A]
print("num_items dist:", Counter(ni).most_common(20))
print("num_items mean", np.mean(ni), "median", np.median(ni))

tp = [a['total_price'] for a in A]
print("total_price examples:", tp[:20])
print("total_price uniq", len(set(tp)), "top:", Counter(tp).most_common(10))
print("empty totals:", sum(1 for x in tp if x.strip()==''))

fp = [tuple(sorted(a['fields_present'].items())) for a in A]
print("fields dist:", Counter(fp).most_common(10))
for k in ['subtotal_present','service_present','tax_present','etc_present']:
    print(k, np.mean([a['fields_present'][k] for a in A]))

G = np.array([a['semantic_grid'] for a in A])
print("grid value dist:", Counter(G.ravel()).most_common())
print("cellwise mode:")
for r in range(4):
    print([Counter(G[:,r*4+c]).most_common(1)[0] for c in range(4)])
gc = Counter([tuple(g) for g in G])
print("uniq grids", len(gc))
for g,c in gc.most_common(10):
    print(c, np.array(g).reshape(4,4).tolist())

print("sizes:", Counter(zip(t.width,t.height)).most_common(5))
print("test sizes:", Counter(zip(te.width,te.height)).most_common(5))
