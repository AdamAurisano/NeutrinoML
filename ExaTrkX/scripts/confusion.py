#!/usr/bin/env python
'''
Script for sparse convolutional network inference
'''
import sys, os

if (len(sys.argv) == 1 or sys.argv[1] == 'config/hit2d.yaml'):
    if '/scratch' not in sys.path: sys.path.append('/scratch')

elif sys.argv[1] == 'config/cori.yaml':
    if os.environ['SLURM_SUBMIT_DIR'] not in sys.path: sys.path.append(os.environ['SLURM_SUBMIT_DIR'])

from core import utils
import datasets
from core.trainers import Trainer
from torch_geometric.loader import DataLoader

import pandas as pd
import matplotlib as mpl
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml, argparse, logging, math, numpy as np, tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sn

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='/scratch/config/hit2d.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def get_legend(names):
  colour = mpl.cm.get_cmap('tab10')
  return plt.legend(handles=[mpatches.Patch(color=colour(i), label=name) for i, name in enumerate(names)])


def get_frame(c, y, i, names):
  colors = colorscale=px.colors.qualitative.G10
  mask = (y == i)
  df = pd.DataFrame(c[mask], columns=['x','y','z'])
  #df['process'] = p[mask]
  f = go.Scatter3d(x=df.x, y=df.y, z=df.z, mode = 'markers',
                         marker=dict(color=colors[i], size=1), name = names[i] ) #text=df.process)
  return f

def main():
  '''Main function'''
  args = parse_args()
  config = configure(args.config)
  trainer = Trainer(**config['trainer'])

  test_dataset = datasets.test_dataset(**config["data"])
  test_loader = DataLoader(test_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)

  mean, std = test_dataset.load_norm()
  transform = datasets.FeatureNorm(mean.to(trainer.device), std.to(trainer.device))

  trainer.build_model(**config['model'])
  trainer.load_state_dict(trainer.best_params(**config["trainer"]))
  trainer.model.eval()
  batch_size = test_loader.batch_size
  n_batches = int(math.ceil(len(test_loader.dataset)/batch_size))
  t = tqdm.tqdm(enumerate(test_loader),total=n_batches)
  
  colour = mpl.cm.get_cmap('tab10')

  names = config['model']['metric_params']['Graph']['class_names']

  y_true_all = torch.empty([0], device=trainer.device)
  y_pred_all = torch.empty([0], device=trainer.device)

  for i, data in t:
    batch_input  = utils.arrange_data.arrange_graph(data, trainer.device)
    batch_output = trainer.model(transform(batch_input))
    batch_target = utils.arrange_truth.arrange_graph_3d(data)

    y_true_all = torch.cat([y_true_all, batch_target.detach()], dim=0)
    y_pred_all = torch.cat([y_pred_all, batch_output.argmax(dim=1).detach()], dim=0)
    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_input
    del batch_output
    del batch_target

  name = config['trainer']['train_name']
  plotdir = f'plots/{name}'
  if not os.path.exists(plotdir): os.mkdir(plotdir)
  confusion = confusion_matrix(y_true=y_true_all.cpu().numpy(), y_pred=y_pred_all.cpu().numpy(), normalize='true')
  plt.figure(figsize=[8,6])
  torch.save(confusion, f'{plotdir}/confusion.pt')
  sn.heatmap(confusion, xticklabels=names, yticklabels=names, annot=True)
  plt.ylim(0, len(names))
  plt.xlabel('Assigned label')
  plt.ylabel('True label')
  plt.savefig(f'{plotdir}/efficiency1.png')
  plt.clf()
  
  y_true = y_true_all.cpu().numpy()
  y_pred = y_pred_all.cpu().numpy()
  torch.save(y_true, f'{plotdir}/y_true.pt')
  torch.save(y_pred, f'{plotdir}/y_pred.pt')

  purity = np.zeros([len(names), len(names)])
  for i in range(len(names)):
    pred = (y_pred_all == i)
    for j in range(len(names)):
      true = (y_true_all == j)
      purity[i,j] = (pred & true).sum() / pred.sum()
    purity[i,:] /= purity[i,:].sum()
  sn.heatmap(purity, xticklabels=names, yticklabels=names, annot=True)
  plt.ylim(0, len(names))
  plt.xlabel('Assigned label')
  plt.ylabel('True label')
  plt.savefig(f'{plotdir}/purity1.png')
  plt.clf()



if __name__ == '__main__':
  main()

