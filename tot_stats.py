import json, pandas as pd, numpy as np, re
from collections import Counter
D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
t = pd.read_csv(D+r'\train.csv')
A = [json.loads(x) for x in t.answer_json]
tp = [a['total_price'] for a in A]
print("charset:", Counter(''.join(tp)).most_common())
print("len dist:", Counter(len(x) for x in tp).most_common())
pat = [re.sub(r'\d','D',x) for x in tp]
print("patterns:", Counter(pat).most_common(15))
print("has letters:", [x for x in tp if re.search(r'[A-Za-z]',x)][:20])
print("with space:", [x for x in tp if ' ' in x][:20])
# digits count
print("ndigits:", Counter(len(re.sub(r'\D','',x)) for x in tp).most_common())
