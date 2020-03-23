'''
PyTorch implementation of PointNet for LArTPC spacepoint deghosting
'''

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate

from time import time # delete me

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        #print(f'Is pos in CUDA? {pos.is_cuda}')
        #t0 = time()
        idx = fps(pos, batch, ratio=self.ratio)
        #t1 = time()
        #print(f'FPS took {t1-t0} seconds.')
        #t0 = t1
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        #t1 = time()
        #print(f'Radius took {t1-t0} seconds')
        #t0 = t1
        edge_index = torch.stack([col, row], dim=0)
        #t1 = time()
        #print(f'Edge index stacking took {t1-t0} seconds.')
        #t0 = t1
        x = self.conv(x, (pos, pos[idx]), edge_index)
        #t1 = time()
        #print(f'Convolution took {t1-t0} seconds.')
        #t0 = t1
        pos, batch = pos[idx], batch[idx]
        #t1 = time()
        #print(f'Filtering positions took {t1-t0} seconds.')
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip

class PointNet(torch.nn.Module):
    def __init__(self, num_features):#, num_classes):
        super(PointNet, self).__init__()
        self.sa1_module = SAModule(0.01, 20, MLP([num_features + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 40, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128+num_features, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 1)#num_classes)
        self.sig  = torch.nn.Sigmoid()

    def forward(self, data):

        #t0 = time()
        sa0_out = (data.x, data.pos, data.batch)
        #t1 = time()
        #print(f'SA 0 took {t1-t0} seconds.')
        #t0 = t1

        sa1_out = self.sa1_module(*sa0_out)
        #t1 = time()
        #print(f'SA 1 took {t1-t0} seconds.')
        #t0 = t1

        sa2_out = self.sa2_module(*sa1_out)
        #t1 = time()
        #print(f'SA 2 took {t1-t0} seconds.')
        #t0 = t1

        sa3_out = self.sa3_module(*sa2_out)
        #t1 = time()
        #print(f'SA 3 took {t1-t0} seconds.')
        #t0 = t1

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        #t1 = time()
        #print(f'FP 3 took {t1-t0} seconds.')
        #t0 = t1

        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        #t1 = time()
        #print(f'FP 2 took {t1-t0} seconds.')
        #t0 = t1

        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        #t1 = time()
        #print(f'FP 1 took {t1-t0} seconds.')
        #t0 = t1

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return self.sig(x).squeeze(-1)


