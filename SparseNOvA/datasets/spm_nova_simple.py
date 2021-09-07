'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import random
from Core import utils

class SparsePixelMapNOvA(Dataset):
    def __init__(self, topdir, subdir, apply_jitter, normalize_coord, **kwargs):
        '''Initialiser for SparsePixelMapNOvA class'''
        self.files = glob.glob(osp.join(topdir, subdir, "*.pt"))
        self.apply_jitter = apply_jitter
        self.normalize_coord = normalize_coord
        
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
        
        if self.normalize_coord:
            norm = torch.tensor([100, 80]).float()    
            data['xfeats'] = torch.cat([data['xfeats'], data['xcoords'].float() / norm], dim=1) 
            data['yfeats'] = torch.cat([data['yfeats'], data['ycoords'].float() / norm], dim=1) 
        
        return data
