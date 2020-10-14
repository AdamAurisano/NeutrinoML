#!/usr/bin/env python

'''
Script for sparse convolutional network inference
'''

import yaml, argparse, logging, math, numpy as np, tqdm
import models, datasets
import torch, torchvision
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix
import seaborn as sns

from Core.trainers import Trainer

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

def collate_sparse(batch):
  for idx, d in enumerate(batch):
    d['c'] = torch.cat((d['c'], torch.LongTensor(d['c'].shape[0],1).fill_(idx)), dim=1)
  ret = { key: torch.cat([d[key] for d in batch], dim=0) for key in batch[0].keys() }
  return ret

def get_legend(names):
  colour = mpl.cm.get_cmap('tab10')
  return plt.legend(handles=[mpatches.Patch(color=colour(i), label=name) for i, name in enumerate(names)])

def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  plt.style.use('ggplot')
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = Trainer(**config['trainer'])

  fulllen = len(full_dataset)
  tv_num = math.ceil(fulllen*config['data']['t_v_split'])
  splits = np.cumsum([fulllen-tv_num,0,tv_num])

  if config['inference']['max_images'] < splits[2] - splits[1]:
    splits[2] = splits[1] + config['inference']['max_images']

  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  valid_loader = DataLoader(valid_dataset, collate_fn=collate_sparse, **config['data_loader'], shuffle=False)

  trainer.build_model(**config['model'])
  trainer.load_state_dict(**config['inference'])
  trainer.model.eval()
  batch_size = valid_loader.batch_size
  n_batches = int(math.ceil(len(valid_loader.dataset)/batch_size))
  t = tqdm.tqdm(enumerate(valid_loader),total=n_batches)

  true_mu_scores = []
  true_pi_scores = []
  reco_mu_scores = []
  reco_pi_scores = []

  for i, data in t:

    batch_output = trainer.model((data['c'].to(trainer.device), data['x'].to(trainer.device), batch_size))
    batch_target = data['y'].to(batch_output.device)

    # 5 is muon, 6 is pion
    cut_true_mu = (batch_target.argmax(dim=1)==5)
    cut_true_pi = (batch_target.argmax(dim=1)==6)
    true_mu_scores += batch_target[cut_true_mu,5].data.tolist()
    true_pi_scores += batch_target[cut_true_pi,5].data.tolist()
    reco_mu_scores += batch_output[cut_true_mu,5].data.tolist()
    reco_pi_scores += batch_output[cut_true_pi,5].data.tolist()

    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_output
    del batch_target

  plt.hist(reco_mu_scores, bins=np.linspace(0, 1, 51), weights=true_mu_scores, density=True, label='True muon')
  plt.hist(reco_pi_scores, bins=np.linspace(0, 1, 51), density=True, label='True pion')
  plt.ylabel('Number of hits [area normed]')
  plt.xlabel('Muon classification score')
  plt.legend()
  plt.tight_layout()
  plt.savefig('reco_scores.png')
  plt.clf()

  plt.hist(true_mu_scores, bins=np.linspace(0, 1, 51), density=True, label='True muon')
  plt.hist(true_pi_scores, bins=np.linspace(0, 1, 51), density=True, label='True pion')
  plt.ylabel('Number of hits [area normed]')
  plt.xlabel('Muon true score')
  plt.legend()
  plt.tight_layout()
  plt.savefig('true_scores.png')
  plt.clf()

  roc_x = []
  roc_y = []
  for cut in np.linspace(0, 1, 101):
    roc_x.append((reco_pi_scores > cut).sum().item() / len(reco_pi_scores))
    roc_y.append((reco_mu_scores > cut).sum().item() / len(reco_mu_scores))
  plt.plot(roc_x, roc_y)
  plt.xlabel('Fraction of true pion hits classified as muon')
  plt.ylabel('Fraction of true muon hits classified as muon')
  plt.tight_layout()
  plt.savefig('roc_curve.png')

if __name__ == '__main__':
  main()

