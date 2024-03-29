#!/usr/bin/env python

'''
Script for sparse convolutional network training
'''

import os, yaml, argparse, logging, math, numpy as np, sys
if '/scratch' not in sys.path: sys.path.append('/scratch')
from SparseProtoDUNE import datasets
from Core import utils
from Core.trainers import Trainer
import torch, sherpa
from torch.utils.data import DataLoader
import torch.nn.functional as F
import MinkowskiEngine as ME
#from torch.nn import LeakyReLU, ReLU, Mish

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='config/sherpa_3d.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def forward(self, x):
  return x * torch.tanh(F.softplus(x))
  
def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = Trainer(**config['trainer'])

  fulllen = len(full_dataset)
  tv_num = math.ceil(fulllen*config['data']['t_v_split'])
  splits = np.cumsum([fulllen-tv_num,0,tv_num])
  collate = utils.collate_sparse_minkowski

  train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  train_loader = DataLoader(train_dataset, collate_fn=collate, **config['data_loader'], shuffle=True, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, collate_fn=collate, **config['data_loader'], shuffle=False)

  parameters = [sherpa.Continuous('learning_rate',
                                  [1e-5, 1e-1]),
                sherpa.Continuous('weight_decay',
                                  [0.01, 0.1]),
                sherpa.Discrete('unet_depth',
                                [2, 6]),
                sherpa.Choice('activation',
                                  [ 'ReLU', 'LeakyReLU', 'mish' ])]
  alg = sherpa.algorithms.GPyOpt(max_num_trials=50)

  study = sherpa.Study(parameters=parameters,
                       algorithm=alg,
                       lower_is_better=False,
                       dashboard_port=os.environ['SHERPA_PORT'])

  for trial in study:
    config['model']['learning_rate'] = trial.parameters['learning_rate']
    config['model']['weight_decay'] = trial.parameters['weight_decay']
    config['model']['unet_depth'] = trial.parameters['unet_depth']
    config['model']['activation_params']['activation'] = trial.parameters['activation']
    trainer.build_model(**config['model'])
    train_summary = trainer.train(
      train_loader,
      valid_data_loader=valid_loader,
      sherpa_study=study,
      sherpa_trial=trial,
      **config['trainer'])
    study.finalize(trial)

if __name__ == '__main__':
  main()

