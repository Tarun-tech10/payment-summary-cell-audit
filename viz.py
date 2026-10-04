import sys, json, os
import numpy as np, pandas as pd, cv2
from prep import process, D

t = pd.read_csv(D+r'\train.csv')
idx = [int(x) for x in sys.argv[1:]] or [0,1,2,3]
out = r'c:\Users\tarun\AppData\Local\Temp\claude\c--Users-tarun-OneDrive-Desktop-ERIS\21c157a1-9ebf-43b4-a7b6-8a9e02cd49da\scratchpad'
os.makedirs(out, exist_ok=True)
for i in idx:
    r = t.iloc[i]
    a = json.loads(r.answer_json)
    p = os.path.join(D, r.image_path.replace('/', os.sep))
    P = process(p)
    H, W = P['H'], P['W']
    vis = P['img'].copy()
    for L in P['lines']:
        cv2.rectangle(vis, (int(L['x0']), int(L['y0'])), (int(L['x1']), int(L['y1'])), (0,0,255), 2)
    x0,y0,x1,y1 = P['box']
    cv2.rectangle(vis, (x0,y0),(x1,y1),(255,255,0),4)
    G = np.array(a['semantic_grid']).reshape(4, 4)
    col = {0:(180,180,180),1:(0,255,0),2:(255,0,255),3:(0,128,255)}
    for rr in range(4):
        for cc in range(4):
            xx = cc*W//4; yy = rr*H//4
            cv2.rectangle(vis, (xx,yy), (xx+W//4, yy+H//4), col[G[rr,cc]], 3)
            cv2.putText(vis, str(G[rr,cc]), (xx+10, yy+50), 0, 1.5, col[G[rr,cc]], 4)
    cv2.imwrite(os.path.join(out, f'viz_{i}.png'), vis)
    print(i, r.id, 'lines', len(P['lines']), 'items', a['num_items'], 'total', a['total_price'], G.tolist())
