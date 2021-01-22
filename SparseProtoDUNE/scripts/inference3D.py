'''
Script for sparse convolutional network inference
'''
from Core import utils
from Core.trainers import Trainer
from SparseProtoDUNE import datasets
from torch.utils.data import DataLoader

import plotly
import pandas as pd
import matplotlib as mpl
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go, plotly.express as px
import yaml, argparse, logging, math, numpy as np, tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sn

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='/scratch/SparseProtoDUNE/config/sparse_3d.yaml')
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
  full_dataset = datasets.get_dataset(**config['data'])
  trainer = Trainer(**config['trainer'])

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
  
  y_true_all = None
  y_pred_all = None
  colour = mpl.cm.get_cmap('tab10')

  names = config['model']['metric_params']['SemanticSegmentation']['class_names']
  for i, data in t:
    batch_input  = utils.arrange_data.arrange_sparse_minkowski(data,trainer.device)
    batch_output = trainer.model(batch_input)
    batch_target = data['y'].to(batch_output.device)

    # Unpack individual pixel maps from batch
    for j in range (batch_input.coords[:,0].max() +1):
      mask = (batch_input.coords[:,0] == j)
      c = batch_input.coords[mask,1:]
      y_pred = batch_output[mask].argmax(dim=1).cpu()
      y_true = batch_target[mask].argmax(dim=1).cpu()
     
      #proc = np.array(data['p'])
      #p = [] 
      #for k in range(len(y_true)):
      #  p.append(proc[k][y_true[k].item()])
      #p = np.array(p)

      if config['inference']['event_display']:
        n_classes = config['model']['n_classes']
        fig_true = go.Figure()
        fig_pred = go.Figure()
        for k in range(n_classes):
          frame_true = get_frame(c, y_true, k,names)
          frame_pred = get_frame(c, y_pred, k,names)
          #frame_true = get_frame(c, y_true, p, k)
          #frame_pred = get_frame(c, y_pred, p, k)
          fig_true.add_trace(frame_true) 
          fig_pred.add_trace(frame_pred) 
        fig_true.update_traces(showlegend=True)
        fig_pred.update_traces(showlegend=True)
        plotly.offline.plot(fig_true, filename =f'plots/evd/evd_{i}_{j}_true.html', auto_open=False)
        plotly.offline.plot(fig_pred, filename =f'plots/evd/evd_{i}_{j}_pred.html', auto_open=False)
        del fig_true 
        del fig_pred 

    if config['inference']['confusion']:
      if y_true_all is None: y_true_all = batch_target.argmax(dim=1)
      else: y_true_all = torch.cat([y_true_all, batch_target.argmax(dim=1)], dim=0)
      if y_pred_all is None: y_pred_all = batch_output.argmax(dim=1)
      else: y_pred_all = torch.cat([y_pred_all, batch_output.argmax(dim=1)], dim=0)
    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_output
    del batch_target
  if config['inference']['confusion']:
    confusion = confusion_matrix(y_true=y_true_all.cpu().numpy(), y_pred=y_pred_all.cpu().numpy(), normalize='true')
    plt.figure(figsize=[8,6])
    torch.save(confusion, 'plots/confusion.py')
    sn.heatmap(confusion, xticklabels=names, yticklabels=names, annot=True)
    plt.ylim(0, len(names))
    plt.xlabel('Assigned label')
    plt.ylabel('True label')
    plt.savefig(f'plots/confusion1.png')
    plt.clf()
  

if __name__ == '__main__':
  main()

