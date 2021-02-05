'''
PyTorch data structure for sparse pixel maps
'''
from torch.utils.data import Dataset
from glob import glob
from functools import partial
import os, os.path as osp, logging, uproot, torch, multiprocessing as mp, numpy as np
from .SegTruth import SegTruth
from .InstanceTruth import *
from time import time

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
    data = torch.load(self.data_files[idx])
    c = torch.IntTensor(data['c']) 
    x = torch.FloatTensor(data['x'])
    y = torch.FloatTensor(data['y'])
    chtm = torch.FloatTensor(data['chtm'])
    #Mix kaons and hip
   # y = np.array(data['y']) 
   # y = np.hstack((y[:,:2], (y[:,2:3] + y[:,4:5]) ,y[:,3:4], y[:,5:]) )
   # y = torch.FloatTensor(y)
    del data
    return { 'x': x, 'c': c, 'y': y, 'chtm': chtm}
   
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

  def process_file(self, filename, feat_norm, voxel_size=0.3, **kwargs):
    '''Process a single raw input file'''
    f = uproot.open(filename)
    t = f['CVNSparse']
    coords, feats, pix_pdg, pix_id, pix_e, pix_proc = t.arrays(
      ['Coordinates', 'Features', 'PixelPDG', 'PixelTrackID', 'PixelEnergy', 'Process'], library='np', how=tuple)
    uuid = osp.basename(filename)[10:-5]

    # Loop over pixel maps in file
    for idx in range(len(feats)):
        coords[idx] = np.array(coords[idx])
        feats[idx] = np.array(feats[idx])
       # if idx !=4: continue 
      #try:
        # Get per-spacepoint ground truth
        start = time()
        m, y, p  = SegTruth(pix_pdg[idx], pix_id[idx], pix_proc[idx], pix_e[idx])
        #print('m & y', len(m), len(y))
        logging.info(f'Ground truth calculating took {time()-start:.2f} seconds.')

        # Get a unique trackId list 
        trks, unique_tracks, counts = get_track_list(pix_pdg[idx], pix_id[idx], pix_e[idx],coords[idx])

       # Voxelise inputs
       
        coordinates = dict()
        features = dict()
        truth = dict()
        process = dict()
        instanceId = dict()
        centroids = dict()

        # Transform spacepoint positions
        transform = np.array([800, -6.5, 0])
        pos = np.array(coords[idx])[m,:] + transform[None,:]
        w = np.zeros([len(coords[idx]),1], dtype=np.float32)  #trackID/voxel
                                                                    #0 is the label for background (bk: deltar ray,  diffuse, michel
                                                                    # and showers for now) 
        InstanceId = 0
        for jdx in range(len(unique_tracks)):
          if counts[jdx]>45:
            mask = (trks  == unique_tracks[jdx])
            w[mask] = InstanceId
            InstanceId +=1
        
        start = time()
        for sp_pos, sp_proc, sp_feats, sp_truth, sp_tid in zip(pos, p[m,:], np.array(feats[idx])[m,:], y[m,:], w[m]):
          vox = tuple( np.floor(val/voxel_size) for val in sp_pos )
          if not vox in coordinates:
            coordinates[vox] = np.array(vox)
            features[vox] = np.zeros(7)
            features[vox][:3] = sp_feats
            features[vox][3:6] = sp_pos
            features[vox][6] = 1
            truth[vox] = sp_truth
            process[vox] = sp_proc
            instanceId[vox] = np.array(sp_tid) 
          else:
            features[vox][:3] += sp_feats
            features[vox][6] += 1
            truth[vox] += sp_truth

        logging.info(f'Voxelising took {time()-start:.2f} seconds.')

        voxId = [instanceId[key].item() for key in coordinates if truth[key].sum()>0]
        c = torch.IntTensor([np.array(coordinates[key]) for key in coordinates if truth[key].sum()>0])
        x = torch.FloatTensor([np.array(features[key]) for key in coordinates if truth[key].sum()>0])
        norm = np.array(feat_norm)
        x = x * norm[None,:] # Normalise features
        y = torch.FloatTensor([truth[key]/truth[key].sum() for key in coordinates if truth[key].sum()>0])
        p = [process[key] for key in coordinates if truth[key].sum()>0]
        if x.max() > 1: print('Feature greater than one at ', x.argmax())
        
        medoids, chtm, offset =  get_InstanceTruth(c,voxId)
        
        data = { 'c': c, 'x': x.float(), 'y': y, 'p':p, 'voxId': voxId, 'medoids':medoids, 'offset': offset, 'chtm':chtm}
        fname = f'pdune_{uuid}_{idx}.pt'
        logging.info(f'Saving file {fname} with {c.shape[0]} voxels.')
        torch.save(data, f'{self.processed_dir}/{fname}')

      #except:
      #  logging.info(f'Exception occurred during processing of event {idx} in file {filename}! Skipping.')
  def process(self, processes, max_files=None, **kwargs):
    '''Process raw input files'''
    proc = partial(self.process_file, **kwargs)
    if max_files is not None:
      files = self.raw_file_names[:max_files]
     # print(type(files),'  ', len(files))
    else:
      files = self.raw_file_names
    if processes == 1:
      for f in files: proc(f)
    else:
      with mp.Pool(processes=processes) as pool:
        pool.map(proc, files)

