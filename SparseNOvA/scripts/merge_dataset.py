#!/usr/bin/env python

# First we need to correctly set the python environment. This is done by adding the top directory of the repository to the Python path. Once that's done, we can import various packages from inside the repository.
import sys, os.path as osp, yaml, argparse, logging, math, numpy as np, torch, sherpa, logging, tqdm, random, time
sys.path.append('/scratch') # This line is equivalent to doing source scripts/source_me.sh in a bash terminal
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from Core.trainers import Trainer
from glob import glob
from Core import utils
from SparseNOvA import datasets
from Core import models
import awkward as ak

# Most of the training options are set in a configuration YAML file. We're going to load this config, and then the options inside will be passed to the relevent piece of the training framework.
parser = argparse.ArgumentParser('train.py')
parser.add_argument('config', nargs='?', default='/scratch/SparseNOvA/config/sparse_nova_mobilenet.yaml')
with open(parser.parse_args().config) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

# Here we load the dataset and the trainer, which is responsible for building the model and overseeing training. There's a block of code which is responsible for slicing the full dataset up into a training dataset and a validation dataset where jitter is applied to training dataset only.
all_nus = sorted(glob(f'/data/mp5/preselection_cvnmap/nu/*.pt'))
all_cosmics = sorted(glob(f'/data/mp5/preselection_cvnmap/cosmic/*.pt'))

if len(all_cosmics) > int(0.1 * len(all_nus)):
    all_cosmics = all_cosmics[0:int(0.1*len(all_nus))]

fulllen_nu = len(all_nus)
fulllen_cosmic = len(all_cosmics)

all_files = all_nus + all_cosmics
full_dataset = datasets.get_dataset(name='SparsePixelMapNOvA', filelist=all_files, apply_jitter=False)

idx = [ i for i in range(len(full_dataset)) ]
random.shuffle(idx)
nproc = 50

for val in full_dataset[0].values():
  print(val.dtype)
exit()

from multiprocessing import Process, Queue
q = Queue()
def process_chunk(indices, dataset, pid, nproc, q):
  builder = ak.ArrayBuilder()
  fields = [ 'xfeats', 'xcoords', 'yfeats', 'ycoords' ]
  chunk = len(dataset) / nproc
  start = int(pid * chunk)
  end = int((pid+1)*chunk)
  for i in indices[start:end]:
    with builder.record():
      for field in fields:
        builder.field(field).append(dataset[i][field].numpy())
      # truth for event needs to be single-element array rather than
      # single value, so they can be stacked into a batch later
      builder.field('truth').append([dataset[i]['truth'].item()])
  q.put(builder.snapshot())
start = time.time()
pool = [ Process(target=process_chunk, args=(idx, full_dataset, i, nproc, q,)) for i in range(nproc) ]
for p in pool: p.start()
arr = ak.concatenate([ q.get() for i in range(nproc) ])
ak.to_parquet(arr, '/data/mp5/cvnmap.parquet')
for p in pool: p.join()
print(f'done processing data! it took {time.time()-start} seconds.')

