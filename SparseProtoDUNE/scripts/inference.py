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
import seaborn as sn

from training.sparse_trainer import SparseTrainer

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
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = SparseTrainer(**config['trainer'])

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

  y_true_all = None
  y_pred_all = None

  colour = mpl.cm.get_cmap('tab10')

  for i, data in t:
    batch_output = trainer.model((data['c'].to(trainer.device), data['x'].to(trainer.device), batch_size))
    batch_target = data['y'].to(batch_output.device)

    # Unpack individual pixel maps from batch
    for j in range(data['c'][:,2].max()):

      mask = (data['c'][:,2] == j)
      coords = data['c'][mask, :-1]
      y_pred = batch_output[mask].argmax(dim=1)
      y_true = batch_target[mask].argmax(dim=1)
#      c_pred = [ f'c{k}' for k in y_pred ]
#      c_true = [ f'c{k}' for k in y_true ]
      if config['inference']['event_display']:
        scatter = plt.scatter(coords[:,0].cpu().numpy(), coords[:,1].cpu().numpy(),
          c=[colour(k) for k in y_pred.cpu().numpy()], s=2)
        #legend = plt.gca().legend(handles=scatter.legend_elements()[0], loc="lower left", labels=config['model']['class_names'])
        plt.gca().add_artist(get_legend(config['model']['class_names']))
        plt.savefig(f'plots/evd/evd_{i}_{j}_pred.png')
        plt.clf()
        scatter = plt.scatter(coords[:,0].cpu().numpy(), coords[:,1].cpu().numpy(),
          c=[colour(k) for k in y_true.cpu().numpy()], s=2)
        #legend = plt.gca().legend(handles=scatter.legend_elements()[0], loc="lower left", labels=config['model']['class_names'])
        plt.gca().add_artist(get_legend(config['model']['class_names']))
        plt.savefig(f'plots/evd/evd_{i}_{j}_true.png')
        plt.clf()
        #print('There are', mask.sum(), 'pixels in this image.')

    if config['inference']['confusion']:
      if y_true_all is None: y_true_all = batch_target.argmax(dim=1)
      else: y_true_all = torch.cat([y_true_all, batch_target.argmax(dim=1)], dim=0)
      if y_pred_all is None: y_pred_all = batch_output.argmax(dim=1)
      else: y_pred_all = torch.cat([y_pred_all, batch_output.argmax(dim=1)], dim=0)
  if config['inference']['confusion']:
    names = config['model']['class_names']
    confusion = confusion_matrix(y_true=y_true_all.cpu().numpy(), y_pred=y_pred_all.cpu().numpy(), normalize='true')
    plt.figure(figsize=[8,6])
    torch.save(confusion, 'plots/confusion.py')
    sn.heatmap(confusion, xticklabels=names, yticklabels=names, annot=True)
    plt.ylim(0, len(names))
    plt.xlabel('Assigned label')
    plt.ylabel('True label')
    plt.savefig(f'plots/confusion.png')
    plt.clf()

if __name__ == '__main__':
  main()

