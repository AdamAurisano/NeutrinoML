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
  def __init__(self, root, trainfiles, raw, **kwargs):
    self.root = root
    self.trainfiles = trainfiles
    self.data_files = self.processed_file_names
    self.raw = raw 

  @property
  def raw_dir(self):
    return f'{self.root}/{self.raw}'

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
    feat_norm =  torch.tensor([1.429e-3,1.667e-3,1.429e-3]).float()
    x = data['x'][:,:3].float()
    y = data['y'].float()
    #is_primary = data['primary'].float()
    #is_primary = is_primary.reshape(is_primary.shape[0],1)
    y = torch.hstack((y[:,:2],(y[:,2:3]+y[:,8:9]),y[:,3:-2]))
    y = torch.hstack((y[:,:3], (y[:,3:4] + y[:,5:6]), y[:,4:5], y[:,6:]))
    if enable_panoptic_seg == False:
      c = data['c'].int()
      #x = torch.cat((x,c*feat_norm[None,:]),dim=1) 
      del data
      return { 'x': x, 'c': c, 'y': y, 'is_primary': is_primary} 
    else:
      c = data['c'].int() 
      x = torch.cat((x,c*feat_norm[None,:]),dim=1) 
      htm = data['htm'].float()
      offset = data['offsets'].int()
      medoids = data['medoids'].int()
      voxId = data['voxId'].int()
      meta = medoids.shape[0]
      P=data['P'].float()
      #is_primary = data['primary'].float()
      #is_primary = is_primary.reshape(is_primary.shape[0],1)
      del data
      return { 'x': x, 'c': c, 'y': y, 'htm': htm,  'medoids':medoids, 
              'offset':offset, 'voxId':voxId, 'meta':meta, 'P':P} 
      #return { 'x': x, 'c': c, 'y': y, 'htm': htm,  'medoids':medoids, 'offset':offset, 'voxId':voxId, 'meta':meta, 'is_primary': is_primary} 
  
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

  def process_file(self, filename, voxel_size=1, **kwargs):
    '''Process a single raw input file'''
    t = uproot.open(filename)['CVNSparse']
    coords, feats, es_energy, pix_pdg, pix_id, pix_e, pix_proc, pix_end_proc, pix_P = t.arrays(
      ['Coordinates', 'Features', 'EstimatedEnergy', 'PixelPDG', 'PixelTrackID', 'PixelEnergy', 'Process', 'EndProcess','Momentum'], library='np', how=tuple)
    uuid = osp.basename(filename)[10:-5]

    mean = np.array([59.11111069, 78.96726227, 58.77098083])
    sigma = np.array([49.33889008, 63.78802109, 48.79499054])

    # Loop over pixel maps in file
    for idx in range(len(feats)):
        fname = f'pdune_{uuid}_{idx}.pt'
        if fname in self.processed_dir: continue
        
        coords[idx] = np.array(coords[idx])
        feats[idx] = np.array(feats[idx])
        es_energy[idx] = np.array(es_energy[idx])
        start = time()
        m, sp_y, sp_id, sp_P = SegTruth(pix_pdg[idx], pix_id[idx], pix_e[idx], pix_proc[idx], pix_end_proc[idx], pix_P[idx] ) # Get semantic label 
       
        ## Voxelization
        pos = np.floor(coords[idx]/voxel_size)
        import pandas as pd

        # concatenate together the numpy arrays and create a dataframe
        cols_coord  = [ "c_x", "c_y", "c_z"           ]
        cols_label  = [ f"l_{i+1}" for i in range(10) ]
        cols_charge = [ "q_u", "q_v", "q_y"           ]
        cols_es_ener = [ "e_u", "e_v", "e_y"           ]
        cols_P      = ["E","px","py","pz"] 
        df = pd.DataFrame(np.concatenate([pos[m,:], sp_y[m,:], feats[idx][m,:], es_energy[idx][m,:], sp_id[m, None], sp_P[m,:]], axis=1),
                columns=cols_coord + cols_label + cols_charge + cols_es_ener + [ "vox_id" ]+ cols_P)
        df["n_sp"] = 1 # when we voxelise, this becomes number of spacepoints per voxel

        # here we define a dictionary telling pandas how to aggregate each column
        agg_label  = { key: "sum"      for key in cols_label  }
        sum_unique = lambda x: sum(set(x))
        agg_charge = { key: sum_unique for key in cols_charge }
        agg_energy = { key: sum_unique for key in cols_es_ener }
        mode = lambda x: stats.mode(x)[0].item()
        agg_P      = {key: mode for key in cols_P}

        # uncomment the print statements below to see how the dataframe is changing
        #
        # group by voxel coordinates and aggregate down to a single row per voxel
        # print("before voxelisation\n", df, "\n\n")
        df = df.groupby(cols_coord).agg(agg_label | agg_charge | agg_energy  | agg_P | { "n_sp": "sum", "vox_id": mode }).reset_index()
        # print("after voxelisation\n", df, "\n\n")

        # just for fun, let's normalise the truth labels
        dE = torch.tensor(df[cols_label].to_numpy()).float()
        def norm_labels(row):
            labels = row[cols_label]
            row[cols_label] = labels / sum(labels)
            return row
        df = df.apply(norm_labels, axis="columns")
        # print("after label norm\n", df, "\n\n")

        # select columns from dataframe to turn into pixel map tensors
        cols_feat = cols_charge + cols_coord + [ "n_sp" ]
        y = torch.tensor(df[cols_label].to_numpy()).float()
        c = torch.tensor(df[cols_coord].to_numpy()).int()
        x = torch.tensor((df[cols_feat].to_numpy()[:,:3] - mean)/sigma).float() #* np.array(feat_norm)[None,:]).float()
        e = torch.tensor(df[cols_es_ener].to_numpy()).float() #estimated energy
        vox_id = torch.tensor(df["vox_id"].to_numpy()).int()
        p = torch.tensor(df[cols_P].to_numpy()).float()

        ## Get Medoids and offsets 
        medoids, htm, offsets, vox_id = get_InstanceTruth(c, vox_id, y.argmax(dim=1), 8)

        # Save file 
        data = { 'c': c, 'x': x, 'e': e, 'y': y, 'voxId': vox_id, 'medoids':  medoids, 'htm': htm, 'offsets': offsets, 'dE':dE, 'P':p }
        logging.info(f'Processing event took:  {time()-start:.2f} seconds.')
   
        if fname not in self.processed_dir and medoids.shape[0]>0:
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


