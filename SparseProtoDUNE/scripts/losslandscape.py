#!/usr/bin/env python

'''
Script for sparse convolutional network training PROTODUNE
'''

import yaml, argparse, logging, math, numpy as np, sys
if '/scratch' not in sys.path: sys.path.append('/scratch')
from SparseProtoDUNE import datasets
from Core import utils
import torch, loss_landscapes
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
  train_loader = DataLoader(train_dataset, collate_fn=collate, **config['data_loader'], shuffle=True, pin_memory=True)

  trainer.build_model(**config['model'])
  trainer.load_state_dict(**config['inference'])

  STEPS=40

  print(iter(train_loader).__next__())
  x, y = iter(train_loader).__next__())
  metric = loss_landscapes.metrics.Loss(model.loss_func, x, y)
  loss_data_fin = loss_landscapes.random_plane(trainer.model, metric, 10, STEPS, normalization='filter', deepcopy_model=True)
  torch.save(loss_data_fin, 'losslandscape.pt')

if __name__ == '__main__':
  main()

