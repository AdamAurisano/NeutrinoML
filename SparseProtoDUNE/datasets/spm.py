'''
PyTorch data structure for sparse pixel maps
'''
from torch.utils.data import Dataset
from glob import glob
import os, os.path as osp, logging, uproot, torch, multiprocessing as mp, pandas as pd
from .Pixel import *
from time import time

class SparsePixelMap(Dataset):
  def __init__(self, root, **kwargs):
    self.root = root
    self.data_files = self.processed_file_names

  @property
  def raw_dir(self):
    return f'{self.root}/raw'

  @property
  def processed_dir(self):
    return f'{self.root}/processed'

  @property
  def train_dir(self):
    return f'{self.root}/train'

  @property
  def raw_file_names(self):
    ret = []
    for subdir in glob(f'{self.raw_dir}/*'):
      ret += glob(f'{subdir}/*.root')
    return ret

  @property
  def processed_file_names(self):
    return glob(f'{self.processed_dir}/*.pt')

  @property
  def train_file_names(self):
    return glob(f'{self.train_dir}/*.pt')

  def __len__(self):
    return len(self.data_files)
    #return len(self.processed_file_names)

  def get_processed_file(self, idx):
    data = torch.load(self.processed_file_names[idx])
    x = torch.FloatTensor(data['PixelValue'])[:,None]
    c = torch.LongTensor(data['Coordinates'])
    y = torch.FloatTensor(data['GroundTruth'])
    #mask = (y[:,0] == 0)
    return { 'x': x, 'c': c, 'y': y }

  def __getitem__(self, idx):
    data = torch.load(self.data_files[idx])
    x = torch.FloatTensor(data['PixelValue'])[:,None]
    c = torch.LongTensor(data['Coordinates'])
    y = torch.FloatTensor(data['GroundTruth'])
    return { 'x': x, 'c': c, 'y': y }

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

  def process_file(self, filename):
    '''Process a single raw input file'''
    f = uproot.open(filename)
    t = f['CVNSparse']
    coords   = t.array('Coordinates')
    vals     = t.array('Values')
    pix_pdg  = t.array('PixelPDG')
    pix_id   = t.array('PixelTrackID')
    pix_e    = t.array('PixelEnergy')
    pix_proc = t.array('Process')
    uuid = osp.basename(filename)[10:-5]
    #vals = f['CVNSparse'].arrays()

    # Loop over pixel maps in file
    for idx in range(len(vals)):

      try:

        fname_1 = f'{self.processed_dir}/pdune_{uuid}_v1_{idx}.pt'
        fname_2 = f'{self.processed_dir}/pdune_{uuid}_v2_{idx}.pt'
        if osp.exists(fname_1) and osp.exists(fname_2):
          logging.info(f'Files {fname_1} and {fname_2} already exist! Skipping.')
          continue

        #coo       = vals[b'Coordinates'][idx]
        #pdg       = vals[b'PixelPDG'][idx]
        #energy    = vals[b'PixelEnergy'][idx]
        #trackId   = vals[b'PixelTrackID'][idx]
        #pixelval  = vals[b'Values'][idx]
        #parents   = vals[b'ParentsPDGs'][idx]
        #process   = vals[b'Process'][idx]

        set = Pixelmap(coo=coords[idx], pixelval=vals[idx], pdg=pix_pdg[idx],
          trackId=pix_id[idx], process=pix_proc[idx], energy=pix_e[idx])

        Vol1, Vol2 = set.get_volumes()
        clvol1     = Pixel.classifier(Vol1)
        clvol2     = Pixel.classifier(Vol2)
        logging.info(f'Saving file pdune_{uuid}_v1_{idx}.pt with {clvol1["PixelValue"].shape[0]} pixels.')
        torch.save(clvol1, f'{self.processed_dir}/pdune_{uuid}_v1_{idx}.pt')
        logging.info(f'Saving file pdune_{uuid}_v2_{idx}.pt with {clvol2["PixelValue"].shape[0]} pixels.')
        torch.save(clvol2, f'{self.processed_dir}/pdune_{uuid}_v2_{idx}.pt')

      except:
        logging.info(f'Exception occurred during processing of event {idx} in file {filename}! Skipping.')

        #print(f'Ground truth processing time was {time()-start:.2f}')

        #print(f'Loop processing took {time()-start:.2f} seconds.')

        #start = time()

        #keys = ['vals','pix_pdg','pix_e','pix_id','pix_proc']
        #branches = [b'Values',b'PixelPDG',b'PixelEnergy',b'PixelTrackID',b'Process']
        #df = pd.DataFrame({key: vals[branch][idx] for key, branch in zip(keys, branches)})
        #print(type(vals[b'Coordinates'][idx]))
        #df['wire'], df['time'], df['tpc'] = np.array(vals[b'Coordinates'][idx]).T
        #print(df.head())
        #inputs = make_sparse_map(df)

        #torch.save(inputs[0], f'{self.processed_dir}/pdune_{uuid}_v1_{idx}_df.pt')
        #torch.save(inputs[1], f'{self.processed_dir}/pdune_{uuid}_v2_{idx}_df.pt')

        #print(f'DataFrame processing took {time()-start:.2f} seconds.')

  def process(self, processes, max_files=None, **kwargs):
    '''Process raw input files'''
    if max_files is not None:
      files = self.raw_file_names[:max_files]
    else:
        files = self.raw_file_names
    if processes == 1:
      for f in files: self.process_file(f)
    else:
      with mp.Pool(processes=processes) as pool:
        pool.map(self.process_file, files)

