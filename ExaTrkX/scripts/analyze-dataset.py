#!/usr/bin/env python

# Import modules
import sys, os, argparse, yaml, logging, numpy as np, tqdm
import torch, torch_geometric
if '/scratch' not in sys.path: sys.path.append('/scratch')
from ExaTrkX import datasets
from Core.trainers import Trainer
from torch_geometric.data import DataLoader

# Configuration options
def configure(config):
  '''Load input configuration file'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

# Configuration options (overwrite default configuration with your own if you want!)
parser = argparse.ArgumentParser('train.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='/scratch/ExaTrkX/config/hit2d.yaml')
args = parser.parse_args()
config = configure(args.config)

full_dataset = datasets.get_dataset(**config['data'])
trainer = Trainer(**config['trainer'])
fulllen = len(full_dataset)
tv_num = np.ceil(fulllen*config['data']['t_v_split'])
splits = np.cumsum([fulllen-tv_num,0,tv_num])

# Load dataset
train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))

# Generate class weights, or load from file if already generated
data_name = os.path.basename(config['data']['path']) + '.pt'
weights_file = os.path.join('/scratch/ExaTrkX/weights', data_name)
if not os.path.exists(weights_file):
  n_classes = config['model']['out_dim']
  total = 0
  weights = np.zeros(n_classes)
  for data in tqdm.tqdm(train_dataset):
    total += data.y.shape[0]
    for i in range(n_classes):
      weights[i] += (data.y == i).sum()
  weights = float(total) / (float(n_classes) * weights)
  if not os.path.isdir('weights'): os.mkdir('weights')
  torch.save(weights, weights_file)
else:
  print('Loading class weights from', weights_file)
  weights = torch.load(weights_file)

print('class weights:')
class_names = config['model']['metric_params']['Graph']['class_names']
for name, weight in zip(class_names, weights):
  print(f'  {name}: {weight}')
  
#config['model']['loss_params']['b_weight'] = torch.tensor(1).float().to(trainer.device)
#config['model']['loss_params']['c_weight'] = torch.tensor(weights[1:]).float().to(trainer.device)
config['model']['loss_params']['weight'] = torch.tensor(weights).float().to(trainer.device)

print(f"Size of full dataset: {len(full_dataset)} ({len(train_dataset)} train, {len(valid_dataset)} validation)")

nhits = []
nedges = []
nedges_class = [ [] for i in range(4) ]

nhits_mu = []
nhits_shower = []
nhits_hadronic = []

for data in tqdm.tqdm(full_dataset):
  nhits.append(data.x.shape[0])
  nedges.append(data.edge_index.shape[1])
  nec = np.empty(4)
  for i in range(4): nedges_class[i].append((data.y == i).sum())
  nhits_hadronic.append(len(np.unique(data.edge_index[:,(data.y==3)])))
  if nedges_class[1][-1] > 0: nhits_shower.append(len(np.unique(data.edge_index[:,(data.y==1)])))
  if nedges_class[2][-1] > 0: nhits_mu.append(len(np.unique(data.edge_index[:,(data.y==2)])))

nhits = np.array(nhits)
nedges = np.array(nedges)
nedges_class = np.array(nedges_class)

print(f"Mean number of hits per graph: {nhits.mean():.2f}")
print(f"Mean number of edges per graph: {nedges.mean():.2f}")

for count, name in zip(nedges_class, class_names):
  print(f"  {name} edges: {count.mean()} per graph")

nues = (nedges_class[1] > 0)
numus = (nedges_class[2] > 0)

print()
print(f"There are {nues.sum()} nues and {numus.sum()} numus - so {len(full_dataset)-(nues.sum()+numus.sum())} we're unsure on.")

print(f"  mean shower edges per nue: {nedges_class[1,nues].mean()}")
print(f"  mean muon edges per numu: {nedges_class[2,numus].mean()}")

nhits_hadronic = np.array(nhits_hadronic)
nhits_shower = np.array(nhits_shower)
nhits_mu = np.array(nhits_mu)

print()
print(f"there are {nhits_hadronic.mean()} hadronic hits per graph, {nhits_shower.mean()} shower hits per nue and {nhits_mu.mean()} hits per numu.")



