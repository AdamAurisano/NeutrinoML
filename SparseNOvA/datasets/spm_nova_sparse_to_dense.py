# -*- coding: utf-8 -*-
'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import random
import MinkowskiEngine as ME
from Core import utils

class DensePixelMapNOvA(Dataset):
    def __init__(self, filelist, apply_jitter, **kwargs):
        '''Initialiser for SparsePixelMapNOvA class'''
        self.files = filelist
        self.apply_jitter = apply_jitter
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        '''Return training information at provided index'''
        if not 0 <= idx < len(self):
            raise Exception(f'Event number {idx} invalid – must be in range 0 -> {len(self)-1}.')

        data = torch.load(self.files[idx])
        if self.apply_jitter:
            scale = random.gauss(1, 0.1)
            data['xfeats'] *= scale
            data['yfeats'] *= scale

        xview = torch.sparse.FloatTensor(data['xcoords'].long().T, data['xfeats'],
                                         torch.Size([100,80,1])).to_dense().permute(2,0,1)
        yview = torch.sparse.FloatTensor(data['ycoords'].long().T, data['yfeats'],
                                         torch.Size([100,80,1])).to_dense().permute(2,0,1)

        return xview, yview, data['truth']
