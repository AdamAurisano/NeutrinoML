# PyTorch geometric dataset for spacepoint graphs
import os, os.path as osp, logging, time, tqdm, random, torch#, multiprocessing as mp
from glob import glob
from functools import partial
import numpy as np, pandas as pd, uproot, torch
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as tf, torch.multiprocessing as mp
import transforms as dune_tf

class SPGraphDataset(Dataset):
  """Spacepoint graph dataset class for PyTorch geometric"""
  def __init__(self, root, transform=None, **kwargs):
    super(SPGraphDataset, self).__init__(root, transform)

  @property
  def raw_file_names(self):
    return glob(f'{self.root}/raw/*.root')

  @property
  def processed_file_names(self):
    return glob(f'{self.root}/processed/*.pt')

  def __len__(self):
    return len(self.processed_file_names)

  def download(self):
    pass

  def get_best_device(self):
    d = {i:torch.cuda.memory_cached(i) for i in range(torch.cuda.device_count())}
    gpu_id = min(d, key=d.get)
    print(type(gpu_id), gpu_id)
    return torch.cuda.device(gpu_id)

  def chunks(self, files, processes):
    '''Split list of files into chunks for parallelisation'''
    chunk_size = int(np.ceil(len(files)/processes))
    for i in range(0, len(files), chunk_size):
      yield files[i:i+chunk_size] if i + chunk_size < len(files) else files[i:]

  def process(self, neighbours=6, processes=50, dist=2, transform=None, devices=None, **kwargs):

    try:
      mp.set_start_method('spawn')
    except RuntimeError:
      pass

    if transform is not None:
      self.transform = tf.Compose([getattr(dune_tf, tfname)() for tfname in transform])

    self.knn = tf.KNNGraph(k=neighbours)

    procs = []
    files = self.chunks(self.raw_paths, processes)
    for i in range(processes):
      device = f'cuda:{devices[i%len(devices)]}'
      p = mp.Process(target=self.process_chunk, args=(next(files), dist, device))
      print(f'Starting process {i} on device {device}.')
      p.start()
      procs.append(p)
    for p in procs:
      p.join()

    #with mp.Pool(processes=processes) as pool:
    #  process_func = partial(self.process_chunk, **kwargs)
    #  pool.map(process_func, self.raw_paths)

    for i, fname in enumerate(glob(f'{self.root}/processed/*.pt')):
      os.rename(fname, f'{self.root}/processed/graph_{i}.pt')

  def process_chunk(self, files, dist, device):
    '''Process a subset of input files'''
    for f in files:
      self.process_file(f, dist, device)

  def process_file(self, filename, dist, device):
    # Load ROOT file
    f = uproot.open(filename)
    t = f['GraphTree']

    pos = t.array('Position')
    feat = t.array('Features')
    truth = t.array('GroundTruth')
    is_cc = t.array('IsCC')
    nu_energy = t.array('NuEnergy')
    lep_energy = t.array('LepEnergy')

    uuid = osp.basename(filename)[4:-5]

    # loop over graphs
    for i in range(len(pos)):
      processed_filename = osp.join(self.processed_dir, f'data_{uuid}_{i}.pt')
      if osp.exists(processed_filename):
        logging.info(f'Processed file {processed_filename} already exists! Skipping...')
        continue

      # Cast truth into DataFrame due to jagged structure
      try:
        truth_df = pd.DataFrame(truth[i], columns=['part_id', 'true_dist', 'unit_x', 'unit_y', 'unit_z'])
      except:
        truth_df = pd.DataFrame(truth[i], columns=['part_id', 'true_dist'])
      y = np.exp(-(dist*truth_df.true_dist))
      y[(truth_df.true_dist<0)] = 0
      # Fill PyTorch data structure
      data = Data(pos=torch.tensor(pos[i], dtype=torch.float),
          x=torch.tensor(feat[i], dtype=torch.float),
          y=torch.tensor(y, dtype=torch.float),
          is_cc=is_cc[i], nu_energy=nu_energy[i], lep_energy=lep_energy[i])
      # Handle edges with kNN
      data = self.knn(data)
      if self.pre_filter is not None and not self.pre_filter(data):
        continue
      if self.transform is not None:
        data = data.to(device)
        self.transform(data)
      torch.save(data, processed_filename)
      logging.info(f'Saved graph with {data.num_nodes} nodes and {data.num_edges} edges as {processed_filename}')

  def get_range(self, data):
    feat_min = [ data.x[:,i].min().item() for i in range(data.x.size(1))]
    feat_max = [ data.x[:,i].max().item() for i in range(data.x.size(1))]
    pos_min  = [ data.pos[:,i].min().item() for i in range(data.pos.size(1))]
    pos_max  = [ data.pos[:,i].max().item() for i in range(data.pos.size(1))]
    for i, a in enumerate(zip(feat_min, feat_max)):
      print(f'Range for feature {i+1} is {a[0]} -> {a[1]}.')
    for i, a in enumerate(zip(pos_min, pos_max)):
      print(f'Range for position {i+1} is {a[0]} -> {a[1]}.')

  def get(self, idx):
    data = torch.load(osp.join(self.processed_dir, f'graph_{idx}.pt'))
    data['is_cc'] = None
    data['nu_energy'] = None
    data['lep_energy'] = None
    #for key in data.keys:
      #print(key)
      #print(data[key])
      #print(data[key].shape)
      #item = data[key] + cumsum[key]
      #if torch.is_tensor(data[key]):
      #  size = data[key].size(data.__cat_dim__(key, data[key]))
      #else:
      #  size = 1

    return data

