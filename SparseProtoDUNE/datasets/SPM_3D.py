'''
PyTorch data structure for sparse pixel maps
'''

from time import time
from glob import glob
from scipy import stats
from functools import partial
from torch.utils.data import Dataset
import os, os.path as osp, logging, uproot, torch, multiprocessing as mp, numpy as np

#Locals
from .InstanceTruth import *
from .SegTruth import SegTruth

class SparsePixelMap3D(Dataset):
  def __init__(self, root, trainfiles, **kwargs):
    self.root = root
    self.trainfiles = trainfiles
    self.data_files = self.processed_file_names

  @property
  def raw_dir(self):
    return f'{self.root}/raw'

  @property
  def processed_dir(self):
    return f'{self.root}/{self.trainfiles}'

  @property
  def raw_file_names(self):
    ret = []
    for subdir in glob(f'{self.raw_dir}/*'):
      ret += glob(f'{subdir}/*.root')
    return ret

  @property
  def processed_file_names(self):
    return glob(f'{self.processed_dir}/*.pt')

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self, idx):
    enable_panoptic_seg = True
    data = torch.load(self.data_files[idx])
    x = torch.tensor(data['x'])
    y = torch.tensor(data['y'])
    if enable_panoptic_seg == False:
      c = torch.tensor(data['c'])
      del data
      return { 'x': x, 'c': c, 'y': y}
    else:
      c = data['c'] 
      htm = data['htm']
      offset = ['offset']
      medoids = ['medoids']
      voxI = data['voxId')
      del data
      return { 'x': x, 'c': c, 'y': y, 'chtm': chtm,  'medoids':medoids, 'offset':offset, 'voxId':voxId} 
    #Mix kaons and hip
   # y = np.array(data['y']) 
   # y = np.hstack((y[:,:2], (y[:,2:3] + y[:,4:5]) ,y[:,3:4], y[:,5:]) )
  
  def vet_files(self):
    for f in self.data_files:
      _, ext = osp.splitext(f)
      if ext != '.pt':
        print('Extension not recognised! Skipping')
        continue
      try:
        torch.load(f)
      except:
        print(f'File {f} is bad! Removing...')
        os.remove(f)

  def process_file(self, filename, feat_norm, voxel_size=1, **kwargs):
    '''Process a single raw input file'''
    t = uproot.open(filename)['CVNSparse']
    coords, feats, pix_pdg, pix_id, pix_e, pix_proc = t.arrays(
      ['Coordinates', 'Features', 'PixelPDG', 'PixelTrackID', 'PixelEnergy', 'Process'], library='np', how=tuple)
    uuid = osp.basename(filename)[10:-5]

    # Loop over pixel maps in file
    for idx in range(len(feats)):
        fname = f'pdune_{uuid}_{idx}.pt'
        if fname in self.processed_dir: continue
        
        coords[idx] = np.array(coords[idx])
        feats[idx] = np.array(feats[idx])
        start = time()
        m, sp_y, sp_id = SegTruth(pix_pdg[idx], pix_id[idx], pix_e[idx]) # Get semantic label 
       
        ## Voxelization
        coordinates = dict()
        features = dict()
        sem_label = dict()

        pos = np.floor(coords[idx]/voxel_size)  
        for vox, sp_truth, sp_feats in zip(pos[m,:], sp_y[m,:], feats[idx][m,:]):
            vox = tuple(vox)
            if not vox in coordinates:
                features[vox] = np.zeros(7)
                coordinates[vox] = vox
                sem_label[vox] = sp_truth
                features[vox][:3] = sp_feats
                features[vox][3:6] = vox
            else:
                features[vox][:3] += sp_feats
                features[vox][6] += 1
                sem_label[vox] += sp_truth
        

        c, vox_id = [], []
        x, y = [], []
        for key in coordinates:
            val = np.array(coordinates[key])
            c.append(val)
            mask = (val == pos)
            mask_vox = mask.sum(axis=1) == 3
            vox_id.append(stats.mode(sp_id[mask_vox])[0].item())
            x.append(features[key])
            y.append(sem_label[key]/(sem_label[key].sum()))
       
       ## Output 
        norm = np.array(feat_norm)
        c = torch.tensor(c)
        y = torch.tensor(y)
        x = torch.tensor(x) * norm[None,:]# Normalise features
        vox_id = torch.tensor(vox_id)
       
        medoids, htm, offsets, vox_id = get_InstanceTruth(c, vox_id, y.argmax(dim=1), 8)
        ## Get Medoids and offsets 
         

        # Save file 
        data = { 'c': c, 'x': x, 'y': y, 'voxId': vox_id, 'medoids':  medoids, 'htm': htm, 'offsets': offsets}
        logging.info(f'Processing event took:  {time()-start:.2f} seconds.')
   
        if fname not in self.processed_dir:
            logging.info(f'Saving file {fname} with {c.shape[0]} voxels.')
            torch.save(data, f'{self.processed_dir}/{fname}')



  def process(self, processes, max_files=None, **kwargs):
    '''Process raw input files'''
    proc = partial(self.process_file, **kwargs)
    if max_files is not None:
      files = self.raw_file_names[:max_files]
    else:
      files = self.raw_file_names
    if processes == 1:
      for f in files: proc(f)
    else:
      with mp.Pool(processes=processes) as pool:
        pool.map(proc, files)


