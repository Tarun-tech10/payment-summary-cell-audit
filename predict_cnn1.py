"""Test-set predictions from the first (70-epoch) CNN fold models, for ensembling."""
import numpy as np, torch
from cnn import Net, to_tensor, W, dev, NI_MAX

Xte = to_tensor(np.load(W + r'\cache\test_img.npy'))
G = np.zeros((len(Xte), 4, 4, 4), np.float32)
N = np.zeros((len(Xte), NI_MAX + 1), np.float32)
F = np.zeros((len(Xte), 4), np.float32)
for k in range(5):
    net = Net().to(dev)
    net.load_state_dict(torch.load(W + f'\\cache\\cnn_f{k}.pt', map_location=dev))
    net.eval()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, len(Xte), 16):
            gl, nl, fl = net(Xte[i:i + 16].to(dev))
            G[i:i + 16] += gl.float().softmax(1).cpu().numpy() / 5
            N[i:i + 16] += nl.float().softmax(1).cpu().numpy() / 5
            F[i:i + 16] += torch.sigmoid(fl.float()).cpu().numpy() / 5
np.save(W + r'\cache\te1_g.npy', G); np.save(W + r'\cache\te1_n.npy', N)
np.save(W + r'\cache\te1_f.npy', F)
print('done', G.shape)
