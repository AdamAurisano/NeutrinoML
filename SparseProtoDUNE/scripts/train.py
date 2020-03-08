#!/usr/bin/env python

'''
Script for sparse convolutional network training
'''

import yaml, argparse, logging, math, numpy as np
import models, datasets
import torch, torchvision
from torch.utils.data import DataLoader

#from training.parallel import ParallelTrainer
#from training.single import SingleTrainer
from training.sparse_trainer import SparseTrainer

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='config/sparse_standard.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def collate_sparse(batch):
  for idx, d in enumerate(batch):
    d['c'] = torch.cat((d['c'], torch.LongTensor(d['c'].shape[0],1).fill_(idx)), dim=1)
  ret = { key: torch.cat([d[key] for d in batch], dim=0) for key in batch[0].keys() }
  return ret

def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = SparseTrainer(**config['trainer'])

  fulllen = len(full_dataset)
  tv_num = math.ceil(fulllen*config['data']['t_v_split'])
  splits = np.cumsum([fulllen-tv_num,0,tv_num])
#  if config['trainer']['max_iters_train'] is not None:
#    train_iters = config['trainer']['max_iters_train']
#    max_train = train_iters * config['data_loader']['batch_size']
#    if splits[0] > max_train: splits[0] = max_train
#  if config['trainer']['max_iters_eval'] is not None:
#    valid_iters = config['trainer']['max_iters_eval']
#    max_valid = valid_iters * config['data_loader']['batch_size']
#    if splits[2] > max_valid: splits[2] = splits[1] + max_valid

  train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  train_loader = DataLoader(train_dataset, collate_fn=collate_sparse, **config['data_loader'], pin_memory=True)
  valid_loader = DataLoader(valid_dataset, collate_fn=collate_sparse, **config['data_loader'], shuffle=False)

  trainer.build_model(**config['model'])

  train_summary = trainer.train(train_loader, valid_data_loader=valid_loader, **config['trainer'])
  print(train_summary)
  torch.save(train_summary, 'summary_test.pt')

if __name__ == '__main__':
  main()

