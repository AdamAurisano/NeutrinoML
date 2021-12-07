#!/usr/bin/env python

# Import modules
import sys, yaml, logging, numpy as np, tqdm
import torch, torch_geometric
if '/scratch' not in sys.path: sys.path.append('/scratch')
from ExaTrkX import datasets, models
from ExaTrkX.trainers.gnn_parallel import GNNParallelTrainer
from torch_geometric.data import DataListLoader, DataLoader

# Configuration options
def configure(config):
  '''Load input configuration file'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

# Configuration options (overwrite default configuration with your own if you want!)
config = configure('/scratch/ExaTrkX/config/hit2d.yaml')

full_dataset = datasets.get_dataset(**config['data'])
device = torch.device(f'cuda:{config["model"]["gpus"][0]}' if torch.cuda.is_available() else 'cpu')
trainer = GNNParallelTrainer(output_dir='./test', device=device, summary_dir=config['trainer']['summary_dir'])

fulllen = len(full_dataset)
tv_num = np.ceil(fulllen*config['data']['t_v_split'])
splits = np.cumsum([fulllen-tv_num,0,tv_num])

# Load dataset
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
loader = DataListLoader if len(config['model']['gpus']) > 1 else DataLoader
valid_loader = loader(valid_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)

# Build model
trainer.build_model(**config['model'])
trainer.load_state_dict(config['test']['state_dict'])

# Test
trainer.draw_output(valid_loader)

