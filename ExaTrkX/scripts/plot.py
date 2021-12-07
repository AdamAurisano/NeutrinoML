#!/usr/bin/env python

# Import modules
import os, sys, yaml, argparse, logging, numpy as np, tqdm
import matplotlib.pyplot as plt
import torch

if (len(sys.argv) == 1 or sys.argv[1] == 'config/hit2d.yaml'):
    if '/scratch' not in sys.path: sys.path.append('/scratch')

elif sys.argv[1] == 'config/cori.yaml':
    if os.environ['SLURM_SUBMIT_DIR'] not in sys.path: sys.path.append(os.environ['SLURM_SUBMIT_DIR'])

import datasets, numl
from core.trainers import Trainer

# Configuration options
def configure(config):
  '''Load input configuration file'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

# Configuration options (overwrite default configuration with your own if you want!)
parser = argparse.ArgumentParser('plot.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='/scratch/config/hit2d.yaml')
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

# Draw plots
name = config['trainer']['train_name']
plotdir = f'plots/{name}'
if not os.path.exists(plotdir): os.makedirs(plotdir)
for i in tqdm.tqdm(range(config['inference']['max_inputs'])):
  graph = valid_dataset[i]
  numl.plot.graph.plot_node_score(graph, graph.y)
  plt.savefig(f'{plotdir}/graph_{i:04d}_node_truth.pdf')
  plt.close()
  numl.plot.graph.plot_edge_score(graph, graph.y_edge)
  plt.savefig(f'{plotdir}/graph_{i:04d}_edge_truth.pdf')
  plt.close()
  y_pred = trainer.model(graph.to(trainer.device)).cpu()
  numl.plot.graph.plot_node_score(graph.cpu(), y_pred.argmax(dim=1))
  plt.savefig(f'{plotdir}/graph_{i:04d}_node_score.pdf')
  plt.close()

