'''
Script for sparse convolutional network inference
'''
import sys
sys.path.append('/scratch')
from Core import utils
from glob import glob
from Core.trainers import Trainer
from SparseNOvA import datasets
from torch.utils.data import DataLoader

import plotly
import pandas as pd
import os.path as osp
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
  add_arg('config', nargs='?', default='config/sparse_nova_dense_mobilenet-Copy1.yaml')
  return parser.parse_args()

def configure(config):
  '''Load configuration'''
  with open(config) as f:
    return yaml.load(f, Loader=yaml.FullLoader)

def get_legend(names):
  colour = mpl.cm.get_cmap('tab10')
  return plt.legend(handles=[mpatches.Patch(color=colour(i), label=name) for i, name in enumerate(names)])


def get_frame(c, y, i, names):
  colors = colorscale=px.colors.qualitative.Vivid
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
  all_nus = sorted(glob(f'{config["data"]["filedir"]}/nu/*.pt'))
  all_cosmics = sorted(glob(f'{config["data"]["filedir"]}/cosmic/*.pt'))
  
  if len(all_cosmics) > int(0.1 * len(all_nus)):
    all_cosmics = all_cosmics[0:int(0.1*len(all_nus))]

  fulllen_nu = len(all_nus)
  fulllen_cosmic = len(all_cosmics)

  tv_num_nu = math.ceil(fulllen_nu*config['data']['t_v_split'])
  tv_num_cosmic = math.ceil(fulllen_cosmic*config['data']['t_v_split'])
    
  splits_nu = np.cumsum([fulllen_nu - tv_num_nu, 0, tv_num_nu])
  splits_cos = np.cumsum([fulllen_cosmic - tv_num_cosmic, 0, tv_num_cosmic])
    
  valid_files = all_nus[splits_nu[1]:splits_nu[2]] + all_cosmics[splits_cos[1]:splits_cos[2]]
  valid_files.sort(key = lambda x: osp.basename(x))
  valid_dataset = datasets.get_dataset(filelist=valid_files, apply_jitter=False, **config['data'])

  trainer = Trainer(**config['trainer'])

  collate = utils.collate_sparse_minkowski_2stack

  valid_loader = DataLoader(valid_dataset, collate_fn=collate, **config['data_loader'], shuffle=False)

  if config['inference']['max_images'] < splits_nu[2] - splits_nu[1]:
    splits_nu[2] = splits_nu[1] + config['inference']['max_images']

  trainer.build_model(**config['model'])
  trainer.load_state_dict(**config['inference'])
  trainer.model.eval()

  batch_size = valid_loader.batch_size
  n_batches = int(math.ceil(len(valid_loader.dataset)/batch_size))
  t = tqdm.tqdm(enumerate(valid_loader),total=n_batches)
  
  y_true_all = None
  y_pred_all = None
  colour = mpl.cm.get_cmap('tab10')

  names = config['model']['metric_params']['Classification']['class_names']
  for i, data in t:
    batch_input  = utils.arrange_data.arrange_dense_2stack(data,trainer.device)
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
    torch.save(confusion, 'plots/Confusion_Matrices/Confusion_FishNet_SGD_Mish.py')
    sn.heatmap(confusion, xticklabels=names, yticklabels=names, annot=True)
    plt.ylim(0, len(names))
    plt.xlabel('Assigned label')
    plt.ylabel('True label')
    plt.savefig(f'plots/Confusion_Matrices/Confusion_FishNet_SGD_Mish.py')
    plt.clf()
    per = { 'matrix': confusion}   
    logging.info(f'Saving matrix')
    torch.save(per, 'plots/Confusion_Matrices/Confusion_FishNet_SGD_Mish.pt' )
if __name__ == '__main__':
  main()
