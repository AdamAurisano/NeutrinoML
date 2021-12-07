#!/usr/bin/env python

# First we need to correctly set the python environment. This is done by adding the top directory of the repository to the Python path. Once that's done, we can import various packages from inside the repository.
import sys, os.path as osp, yaml, argparse, logging, math, numpy as np, torch, sherpa, logging, tqdm, random, time
sys.path.append('/scratch') # This line is equivalent to doing source scripts/source_me.sh in a bash terminal
from torch.utils.data import DataLoader
from Core import utils
from ExaTrkX import datasets
import awkward as ak

# Most of the training options are set in a configuration YAML file. We're going to load this config, and then the options inside will be passed to the relevent piece of the training framework.
parser = argparse.ArgumentParser('train.py')
parser.add_argument('config', nargs='?', default='/scratch/ExaTrkX/config/hit2d.yaml')
with open(parser.parse_args().config) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

full_dataset = datasets.get_dataset(**config['data'])

nproc = 50

from multiprocessing import Process, Queue
q = Queue()
def process_chunk(dataset, pid, nproc, q):
  builder = ak.ArrayBuilder()
  chunk = len(dataset) / nproc
  start = int(pid * chunk)
  end = int((pid+1)*chunk)
  for i in range(start, end):
    with builder.record():
      for key, val in dataset[i]:
        builder.field(key).append(val.numpy())
  q.put(builder.snapshot())
start = time.time()
pool = [ Process(target=process_chunk, args=(full_dataset, i, nproc, q,)) for i in range(nproc) ]
for p in pool: p.start()
arr = ak.concatenate([ q.get() for i in range(nproc) ])
ak.to_parquet(arr, '/data/hit2d/flav-shower.parquet')
for p in pool: p.join()
print(f'done processing data! it took {time.time()-start} seconds.')

