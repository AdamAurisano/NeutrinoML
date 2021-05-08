# -*- coding: utf-8 -*-
'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import awkward as ak
import MinkowskiEngine as ME
from Core import utils

class DensePixelMapNOvA(Dataset):
    def __init__(self, filelist, apply_jitter, **kwargs):
        '''Initialiser for SparsePixelMapNOvA class'''
        self.data = ak.from_parquet('/data/mp5/cvnmap.parquet', lazy=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''Return training information at provided index'''
        if not 0 <= idx < len(self):
            raise Exception(f'Event number {idx} invalid – must be in range 0 -> {len(self)-1}.')

        data = { field: torch.from_numpy(ak.to_numpy(self.data[idx][field])) for field in self.data.fields }
        data['xfeats'] = data['xfeats'].type(torch.float32)
        data['xcoords'] = data['xcoords'].type(torch.int32)
        data['yfeats'] = data['yfeats'].type(torch.float32)
        data['ycoords'] = data['ycoords'].type(torch.int32)

        xview = torch.sparse.FloatTensor(data['xcoords'].long().T, data['xfeats'],
                                         torch.Size([100,80,1])).to_dense().permute(2,0,1)
        yview = torch.sparse.FloatTensor(data['ycoords'].long().T, data['yfeats'],
                                         torch.Size([100,80,1])).to_dense().permute(2,0,1)

        return xview, yview, data['truth']
