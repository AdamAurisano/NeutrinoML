'''Simple PyTorch data loader'''

from torch.utils.data import Dataset
import os.path as osp, glob, h5py, tqdm, numpy as np, torch

class SimpleDataset(Dataset):
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
        return torch.load(self.files[idx])
