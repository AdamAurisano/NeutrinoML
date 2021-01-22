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
  fulllen_nu = len(all_nus)
  tv_num_nu = math.ceil(fulllen_nu*config['data']['t_v_split'])
  splits_nu = np.cumsum([fulllen_nu - tv_num_nu, 0, tv_num_nu])

  valid_files = all_nus[splits_nu[1]:splits_nu[2]]
  valid_files.sort(key = lambda x: osp.basename(x))
  valid_dataset = datasets.get_dataset(filelist=valid_files, apply_jitter=False, **config['data'])
  trainer = Trainer(**config['trainer'])
  collate = getattr(utils, config['model']['collate'])
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

  names = config['model']['metric_params']['Classification']['class_names']
  score_numu = []
  score_nue = []
  score_not_numu = []
  score_not_nue = []
  for i, data in t:
    batch_input  = utils.arrange_data.arrange_dense_2stack(data,trainer.device)
    batch_output = trainer.model(batch_input)
    batch_target = data['y'].to(batch_output.device)
    
    true_numu = (batch_target == 0)
    true_nue = (batch_target == 1)
    # score of # of correctly classified NuMu
    score_numu += batch_output[true_numu, 0].data.tolist()
    # score of # of correctly classified NuE
    score_nue += batch_output[true_nue, 1].data.tolist() 
    # score of # of incorrectly classified NuMu which are now NuE
    score_not_numu += batch_output[true_nue, 0].data.tolist()
    # score of # of incorrectly classified NuE whcih are now NuMu
    score_not_nue += batch_output[true_numu, 1].data.tolist()

  plt.hist(score_not_numu, bins=np.linspace(0, 1, 51), density=True, label='Not NuMu')
  plt.xlabel('Incorrect NuMu CC Classifier Output')
  plt.ylabel('Events')
  plt.savefig('plots/Dense_MobileNet_SGD_Mish_StepLR/Incorrect_NuMu_distribution.png')
  plt.clf()
  plt.hist(score_not_nue, bins=np.linspace(0, 1, 51), density=True, label='Not NuE')
  plt.xlabel('Incorrect NuE CC Classifier Output')
  plt.ylabel('Events')
  plt.savefig('plots/Dense_MobileNet_SGD_Mish_StepLR/Incorrect_NuE_distribution.png')
  plt.clf()
  plt.hist(score_nue, bins=np.linspace(0, 1, 51), density=True, label='NuE')
  plt.xlabel('NuE CC Classifier Output')
  plt.ylabel('Events')
  plt.savefig('plots/Dense_MobileNet_SGD_Mish_StepLR/NuE_distribution.png')
  plt.clf()
  plt.hist(score_numu, bins=np.linspace(0, 1, 51), density=True, label='NuMu')
  plt.xlabel('NuMu CC Classifier Output')
  plt.ylabel('Events')
  plt.savefig('plots/Dense_MobileNet_SGD_Mish_StepLR/NuMu_distribution.png')
  plt.clf()

if __name__ == '__main__':
  main()
