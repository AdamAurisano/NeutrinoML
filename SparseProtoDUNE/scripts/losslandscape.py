#!/usr/bin/env python

'''
Script for sparse convolutional network training PROTODUNE
'''

import yaml, argparse, logging, math, numpy as np, sys
if '/scratch' not in sys.path: sys.path.append('/scratch')
from SparseProtoDUNE import datasets
from Core import utils
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='/scratch/SparseProtoDUNE/config/sparse_3d.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  full_dataset = datasets.get_dataset(**config['data'])
  if config['model']['instance_segmentation']:
    from Core.trainers.trainerInsSeg import TrainerInsSeg
    trainer = TrainerInsSeg(**config['trainer'])
    collate = utils.collate_sparse_minkowski_panoptic
  else:
    from Core.trainers.trainer import Trainer
    trainer = Trainer(**config['trainer'])
    collate = utils.collate_sparse_minkowski

  print(collate)
  fulllen = len(full_dataset)
  tv_num = math.ceil(fulllen*config['data']['t_v_split'])
  splits = np.cumsum([fulllen-tv_num,0,tv_num])

  train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  train_loader = DataLoader(train_dataset, collate_fn=collate, **config['data_loader'], shuffle=True, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, collate_fn=collate, batch_size=1, shuffle=False)

  trainer.build_model(**config['model'])
  train_summary = trainer.train(train_loader, valid_data_loader=valid_loader, **config['trainer'])
  print(train_summary)
  torch.save(train_summary, 'summary_test.pt')

if __name__ == '__main__':
  main()

