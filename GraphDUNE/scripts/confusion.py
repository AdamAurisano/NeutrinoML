'''
Script for sparse convolutional network inference
'''
import sys
if '/scratch' not in sys.path: sys.path.append('/scratch')
from Core import utils
from GraphDUNE import datasets
from Core.trainers import Trainer
from torch_geometric.data import DataLoader

import pandas as pd
import matplotlib as mpl
import torch, torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml, argparse, logging, math, numpy as np, tqdm

from sklearn.metrics import confusion_matrix
import seaborn as sn

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='/scratch/GraphDUNE/config/hit2d.yaml')
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

  valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
  valid_loader = DataLoader(valid_dataset, batch_size=config['trainer']['batch_size'], shuffle=False)

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
    batch_input  = utils.arrange_data.arrange_graph(data,trainer.device)
    batch_output = trainer.model(batch_input)
    batch_target = data['y'].to(batch_output.device)

      #proc = np.array(data['p'])
      #p = [] 
      #for k in range(len(y_true)):
      #  p.append(proc[k][y_true[k].item()])
      #p = np.array(p)

    if y_true_all is None: y_true_all = batch_target.cpu()
    else: y_true_all = torch.cat([y_true_all, batch_target.cpu()], dim=0)
    if y_pred_all is None: y_pred_all = batch_output.argmax(dim=1).cpu()
    else: y_pred_all = torch.cat([y_pred_all, batch_output.argmax(dim=1).cpu()], dim=0)
    # PyTorch doesn't delete these automatically for some reason, so do it manually
    del batch_input
    del batch_output
    del batch_target



  confusion = confusion_matrix(y_true=y_true_all.cpu().numpy(), y_pred=y_pred_all.cpu().numpy(), normalize='true')
  plt.figure(figsize=[8,6])
  torch.save(confusion, 'plots/confusion.pt')
  sn.heatmap(confusion, xticklabels=names, yticklabels=names, annot=True)
  plt.ylim(0, len(names))
  plt.xlabel('Assigned label')
  plt.ylabel('True label')
  plt.savefig(f'plots/confusion1.png')
  plt.clf()
  

if __name__ == '__main__':
  main()

