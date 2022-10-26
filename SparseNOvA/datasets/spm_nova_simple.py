'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch
import random
from Core import utils

class SparsePixelMapNOvA(Dataset):
    def __init__(self, topdir, subdir, apply_jitter, standardize_input, limit=None, **kwargs):
    # def __init__(self, topdir, subdir, limit=None, **kwargs):
        '''Initialiser for SparsePixelMapNOvA class'''
        self.files = glob.glob(osp.join(topdir, subdir, "*.pt"))
        if limit and len(self.files) > limit:
            self.files = self.files[0:limit]
        self.apply_jitter = apply_jitter
        # self.normalize_coord = normalize_coord
        self.standardize_input = standardize_input
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        '''Return training information at provided index'''
        if not 0 <= idx < len(self):
            raise Exception(f'Event number {idx} invalid – must be in range 0 -> {len(self)-1}.')

        data = torch.load(self.files[idx])
        ret = {}
        
        if self.apply_jitter:
            scale = random.gauss(1, 0.1)
            data['xfeats'] *= scale
            data['yfeats'] *= scale
        
        # if self.normalize_coord:
        #     norm = torch.tensor([100, 80]).float()    
        #     data['xfeats'] = torch.cat([data['xfeats'], data['xcoords'].float() / norm], dim=1) 
        #     data['yfeats'] = torch.cat([data['yfeats'], data['ycoords'].float() / norm], dim=1) 
            
        if self.standardize_input:
            mean_x = 14.5679
            std_x = 16.2710
            mean_y = 14.5137
            std_y = 16.2276
            norm = torch.tensor([100, 80]).float()
            
            ret = {'xfeats': torch.cat([((data['xfeats'].float() - mean_x) / std_x), data['xcoords'].float() / norm], dim=1),
                   'yfeats': torch.cat([((data['yfeats'].float() - mean_y) / std_y), data['ycoords'].float() / norm], dim=1),
                   'xcoords': data['xcoords'],
                   'ycoords': data['ycoords'],
                   'xsegtruth': data['xsegtruth'],
                   'ysegtruth': data['ysegtruth'],
                   'xinstruth': data['xinstruth'],
                   'yinstruth': data['yinstruth'],
                   'evttruth': data['evttruth']}
            
        
        del data
        
        return ret
