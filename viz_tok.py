import os, json, sys
import numpy as np, pandas as pd, cv2
from ocr_extract import extract, D
OUT = r'c:\Users\tarun\AppData\Local\Temp\claude\c--Users-tarun-OneDrive-Desktop-ERIS\21c157a1-9ebf-43b4-a7b6-8a9e02cd49da\scratchpad'
t = pd.read_csv(D+r'\train.csv'); A=[json.loads(x) for x in t.answer_json]
for i in [int(a) for a in sys.argv[1:]]:
    p = os.path.join(D, t.image_path[i].replace('/', os.sep))
    E = extract(p)
    vis = E['img'].copy()
    for L in E['lines']:
        for tk in L['tokens']:
            cv2.rectangle(vis,(int(tk['x0']),int(tk['y0'])),(int(tk['x1']),int(tk['y1'])),(0,140,255),1)
            for (x,y,w,h,a) in tk['boxes']:
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(vis,str(len(tk['boxes'])),(int(tk['x0']),int(tk['y0'])-2),0,0.35,(255,0,0),1)
    cv2.imwrite(os.path.join(OUT,f'tok_{i}.png'), vis)
    print(i, 'ch', round(E['ch'],1), 'lines', len(E['lines']), 'total', A[i]['total_price'])
