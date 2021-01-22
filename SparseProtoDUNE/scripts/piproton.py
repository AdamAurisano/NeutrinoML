#!/usr/bin/env python

'''
Script for sparse convolutional network inference
'''

from Core import utils
from Core.trainers import SparseTrainer
from SparseProtoDUNE import datasets
from torch.utils.data import DataLoader

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml, argparse, logging, math, numpy as np, tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sns


def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='config/sparse_3d.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)


def get_legend(names):
  colour = mpl.cm.get_cmap('tab10')
  return plt.legend(handles=[mpatches.Patch(color=colour(i), label=name) for i, name in enumerate(names)])

def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  plt.style.use('ggplot')
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = SparseTrainer(**config['trainer'])

  fulllen = len(full_dataset)
  tv_num = math.ceil(fulllen*config['data']['t_v_split'])
  splits = np.cumsum([fulllen-tv_num,0,tv_num])
  collate = utils.collate_sparse_minkowski

  if config['inference']['max_images'] < splits[2] - splits[1]:
    splits[2] = splits[1] + config['inference']['max_images']

  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  valid_loader = DataLoader(valid_dataset, collate_fn=collate, **config['data_loader'], shuffle=False)

  trainer.build_model(**config['model'])
  trainer.load_state_dict(**config['inference'])
  trainer.model.eval()
  batch_size = valid_loader.batch_size
  n_batches = int(math.ceil(len(valid_loader.dataset)/batch_size))
  t = tqdm.tqdm(enumerate(valid_loader),total=n_batches)

  true_hip_scores = []
  true_pi_scores = []
  reco_hip_scores = []
  reco_pi_scores = []

  for i, data in t:
    batch_input  = utils.arrange_data.arrange_sparse_minkowski(data,trainer.device)
    batch_output = trainer.model(batch_input)
    batch_target = data['y'].to(batch_output.device)

    # 3d -> n1 =  7 is pion,  n2 = 5 is hip
    n1 = 7 
    n2 = 5
    cut_true_pi = (batch_target.argmax(dim=1)==n1)
    cut_true_hip = (batch_target.argmax(dim=1)==n2)
    true_pi_scores += batch_target[cut_true_pi,n1].data.tolist()
    true_hip_scores += batch_target[cut_true_hip,n1].data.tolist()
    reco_pi_scores += batch_output[cut_true_pi,n1].data.tolist()
    reco_hip_scores += batch_output[cut_true_hip,n1].data.tolist()

    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_output
    del batch_target

  plt.hist(reco_pi_scores, bins=np.linspace(0, 1, 51), weights=true_pi_scores, density=True, label='True pions', alpha = 0.5)
  plt.hist(reco_hip_scores, bins=np.linspace(0, 1, 51), density=True, label='True protons', alpha = 0.5)
  plt.ylabel('Number of hits [area normed]')
  plt.xlabel('Pion classification score')
  plt.legend()
  plt.tight_layout()
  plt.savefig('plots/piproton_reco_scores.png')
  plt.clf()

  plt.hist(true_pi_scores, bins=np.linspace(0, 1, 51), density=True, label='True pion')
  plt.hist(true_hip_scores, bins=np.linspace(0, 1, 51), density=True, label='True protons')
  plt.ylabel('Number of hits [area normed]')
  plt.xlabel('Pion true score')
  plt.legend()
  plt.tight_layout()
  plt.savefig('plots/piproton_true_scores.png')
  plt.clf()

  roc_x = []
  roc_y = []
  for cut in np.linspace(0, 1, 101):
    roc_x.append((reco_hip_scores > cut).sum().item() / len(reco_hip_scores))
    roc_y.append((reco_pi_scores > cut).sum().item() / len(reco_pi_scores))
  plt.plot(roc_x, roc_y)
  plt.xlabel('Fraction of true proton hits classified as pions')
  plt.ylabel('Fraction of true pion hits classified as pions')
  plt.tight_layout()
  plt.savefig('plots/piproton_roc_curve.png')

if __name__ == '__main__':
  main()

