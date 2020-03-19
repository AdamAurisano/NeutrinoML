'''
Script for sparse convolutional network inference
'''
import pandas as pd
import plotly.graph_objects as go, plotly.express as px
import plotly
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
    #print(batch_target.shape)
    #print(data['c'][:,3].max())
    # Unpack individual pixel maps from batch

    for j in range(data['c'][:,3].max()):
      
      mask = (data['c'][:,3] == j)
      #coords = data['c'][mask, :-1]
      y_pred = batch_output[mask].argmax(dim=1)
      y_true = batch_target[mask].argmax(dim=1)
     # df = pd.DataFrame(data['x'][mask].cpu().numpy()[:,3:6], columns=['x', 'y', 'z'])
      df = pd.DataFrame(data['c'][mask, :-1].cpu().numpy(), columns=['x', 'y', 'z'])
      df['truth'] = y_true.cpu().numpy()
      df['pred'] = y_pred.cpu().numpy()
#      c_pred = [ f'c{k}' for k in y_pred ]
#      c_true = [ f'c{k}' for k in y_true ]
      if config['inference']['event_display']:
        #true
        fig_true = go.Figure(data=[go.Scatter3d(x=df.x, y=df.y, z=df.z, mode='markers', 
            marker=dict(color=df.truth, size=1, colorscale=px.colors.qualitative.G10))])
        plotly.offline.plot(fig_true, filename =f'plots/evd/evd_{i}_{j}_true.html', auto_open=False)
        fig = plt.figure(figsize=(8,5))
        plt.gca().add_artist(get_legend(config['model']['class_names']))
        plt.savefig(f'plots/evd/evd_{i}_{j}_true_Legend.png')      
        fig.clf()
	#prediction
        fig_pred = go.Figure(data=[go.Scatter3d(x=df.x, y=df.y, z=df.z, mode='markers', 
            marker=dict(color=df.pred, size=1, colorscale=px.colors.qualitative.G10))])
        plotly.offline.plot(fig_pred, filename = f'plots/evd/evd_{i}_{j}_pred.html', auto_open=False)
        fig = plt.figure(figsize=(8,5))
        plt.gca().add_artist(get_legend(config['model']['class_names']))
        plt.savefig(f'plots/evd/evd_{i}_{j}_pred_Legend.png')      
        fig.clf()

    if config['inference']['confusion']:
      if y_true_all is None: y_true_all = batch_target.argmax(dim=1)
      else: y_true_all = torch.cat([y_true_all, batch_target.argmax(dim=1)], dim=0)
      if y_pred_all is None: y_pred_all = batch_output.argmax(dim=1)
      else: y_pred_all = torch.cat([y_pred_all, batch_output.argmax(dim=1)], dim=0)
    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_output
    del batch_target
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

