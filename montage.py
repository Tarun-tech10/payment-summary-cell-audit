import sys, json, os
import numpy as np, pandas as pd, cv2
from prep import process, D

t = pd.read_csv(D+r'\train.csv')
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
out = r'c:\Users\tarun\AppData\Local\Temp\claude\c--Users-tarun-OneDrive-Desktop-ERIS\21c157a1-9ebf-43b4-a7b6-8a9e02cd49da\scratchpad'
tiles = []
TH, TW = 480, 320
for i in range(start, start+k):
    r = t.iloc[i]
    a = json.loads(r.answer_json)
    P = process(os.path.join(D, r.image_path.replace('/', os.sep)))
    H, W = P['H'], P['W']
    vis = P['img'].copy()
    for L in P['lines']:
        cv2.rectangle(vis, (int(L['x0']), int(L['y0'])), (int(L['x1']), int(L['y1'])), (0,0,255), 3)
    x0,y0,x1,y1 = P['box']
    cv2.rectangle(vis, (x0,y0),(x1,y1),(255,255,0),5)
    G = np.array(a['semantic_grid']).reshape(4,4)
    col = {0:(180,180,180),1:(0,255,0),2:(255,0,255),3:(0,128,255)}
    for rr in range(4):
        for cc in range(4):
            xx=cc*W//4; yy=rr*H//4
            cv2.rectangle(vis,(xx,yy),(xx+W//4,yy+H//4),col[G[rr,cc]],3)
    tiles.append(cv2.resize(vis,(TW,TH)))
    print(i, r.id, 'lines', len(P['lines']), 'items', a['num_items'], 'total', repr(a['total_price']), G.tolist())
rows = [np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]
w = max(x.shape[1] for x in rows)
rows = [np.pad(x,((0,0),(0,w-x.shape[1]),(0,0))) for x in rows]
cv2.imwrite(os.path.join(out,'mont.png'), np.vstack(rows))
