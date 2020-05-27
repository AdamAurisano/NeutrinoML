'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import random
from SparseBase import utils

class SparsePixelMapNOvA(Dataset):
    def __init__(self, filedir, **kwargs):
        '''Initialiser for SparsePixelMapNOvA class'''
        self.filedir = filedir
        self.files = sorted(glob.glob(f'{self.filedir}/*.pt'))
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        '''Return training information at provided index'''
        if not 0 <= idx < len(self):
            raise Exception(f'Event number {idx} invalid – must be in range 0 -> {len(self)-1}.')

        data = torch.load(self.files[idx])
        scale = random.gauss(1, 0.1)
        print(data['xfeats'], data['yfeats'])
        data['xfeats'] *= scale
        data['yfeats'] *= scale
        print(data['xfeats'], data['yfeats'])
        
        return data
#         return torch.load(self.files[idx])
