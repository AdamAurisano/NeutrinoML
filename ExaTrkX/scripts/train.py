#!/usr/bin/env python

# Import modules
def init():
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

  train_loader = DataLoader(train_dataset, batch_size=config['trainer']['batch_size'], shuffle=True, pin_memory=True)
  valid_loader = DataLoader(valid_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)

  return config, trainer, train_loader, valid_loader

def train():
  import torch
  torch.autograd.set_detect_anomaly(True)
  config, trainer, train_loader, valid_loader = init()

  # Build model
  trainer.build_model(**config['model'])

  # Train!
  train_summary = trainer.train(train_loader, config['trainer']['n_epochs'], valid_data_loader=valid_loader)
  print(train_summary)

if __name__ == "__main__":
  train()
