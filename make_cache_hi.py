"""Higher-resolution image cache (H,W divisible by 4 so whole-cell shifts stay exact)."""
import os, numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor

D = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_summary'
OUT = r'c:\Users\tarun\OneDrive\Desktop\ERIS\payment_work\cache'
IH, IW = 640, 448


def make(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    norm = cv2.divide(g, bg, scale=255)
    th = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 21, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return np.concatenate([
        cv2.resize(img, (IW, IH), interpolation=cv2.INTER_AREA),
        cv2.resize(th, (IW, IH), interpolation=cv2.INTER_AREA)[..., None],
        cv2.resize(norm, (IW, IH), interpolation=cv2.INTER_AREA)[..., None]], axis=2)


def run(df, name):
    paths = [os.path.join(D, p.replace('/', os.sep)) for p in df.image_path]
    with ThreadPoolExecutor(8) as ex:
        arrs = list(ex.map(make, paths))
    A = np.stack(arrs).astype(np.uint8)
    np.save(os.path.join(OUT, name + '_img_hi.npy'), A)
    print(name, A.shape, flush=True)


if __name__ == '__main__':
    run(pd.read_csv(D + r'\train.csv'), 'train')
    run(pd.read_csv(D + r'\test.csv'), 'test')
