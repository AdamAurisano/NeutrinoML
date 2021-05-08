'''
PyTorch data structure for sparse pixel maps
'''

from torch.utils.data import Dataset
import numpy as np, torch, awkward as ak
from Core import utils

class SparsePixelMapNOvA(Dataset):
  def __init__(self, filename, **kwargs):
    '''Initialiser for SparsePixelMapNOvA class'''
    self.data = ak.from_parquet('/data/mp5/cvnmap.parquet', lazy=True)
      
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    '''Return training information at provided index'''
    if not 0 <= idx < len(self):
      raise Exception(f'Event number {idx} invalid – must be in range 0 -> {len(self)-1}.')

    ret = { field: torch.from_numpy(ak.to_numpy(self.data[idx][field])) for field in self.data.fields }
    ret['xfeats'] = ret['xfeats'].type(torch.float32)
    ret['xcoords'] = ret['xcoords'].type(torch.int32)
    ret['yfeats'] = ret['yfeats'].type(torch.float32)
    ret['ycoords'] = ret['ycoords'].type(torch.int32)
    return ret

