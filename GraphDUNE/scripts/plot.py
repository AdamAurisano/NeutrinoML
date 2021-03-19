#!/usr/bin/env python

# Import modules
import os, sys, yaml, argparse, logging, numpy as np, tqdm
import matplotlib.pyplot as plt
import torch, torch_geometric
if '/scratch' not in sys.path: sys.path.append('/scratch')
from GraphDUNE import datasets
from GraphDUNE.plot import graphplot
from Core.trainers import Trainer
from torch_geometric.data import DataLoader

# Configuration options
def configure(config):
  '''Load input configuration file'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

# Configuration options (overwrite default configuration with your own if you want!)
parser = argparse.ArgumentParser('plot.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='/scratch/GraphDUNE/config/hit2d.yaml')
args = parser.parse_args()
config = configure(args.config)

full_dataset = datasets.get_dataset(**config['data'])
trainer = Trainer(**config['trainer'])

fulllen = len(full_dataset)
tv_num = np.ceil(fulllen*config['data']['t_v_split'])
splits = np.cumsum([fulllen-tv_num,0,tv_num])

# Load dataset
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))

# Build model
trainer.build_model(**config['model'])
trainer.load_state_dict(**config['inference'])

gp = graphplot.GraphPlot()

# Draw plots
name = config['trainer']['train_name']
if not os.path.exists(f'plots/{name}'): os.makedirs(f'plots/{name}')
for i in tqdm.tqdm(range(26)):
  if i != 6 and i != 25: continue
  graph = valid_dataset[i]
  gp.plot_edge_truth(graph)
  if i == 6:
    plt.xlim([7200,7700])
    plt.ylim([7200,8400])
    plt.tight_layout()
  plt.savefig(f'plots/{name}/graph_{i:04d}_truth.pdf')
  plt.close()
  gp.plot_edge_score(trainer, graph)
  if i == 6:
    plt.xlim([7200,7700])
    plt.ylim([7200,8400])
    plt.tight_layout()
  plt.savefig(f'plots/{name}/graph_{i:04d}_score.pdf')
  plt.close()
  gp.plot_edge_diff(trainer, graph)
  plt.savefig(f'plots/{name}/graph_{i:04d}_diff.pdf')
  plt.close()

