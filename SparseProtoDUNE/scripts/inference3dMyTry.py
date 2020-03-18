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

#from training.parallel import ParallelTrainer
#from training.single import SingleTrainer
from training.sparse_trainer import SparseTrainer

def parse_args():
  '''Parse arguments'''
  parser = argparse.ArgumentParser('process.py')
  add_arg = parser.add_argument
  add_arg('config', nargs='?', default='config/sparse_standard.yaml')
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
  #device = torch.device(f'cuda:{config["model"]["gpus"][0]}' if torch.cuda.is_available() else 'cpu')
  #trainer = ParallelTrainer(output_dir='./test', device=device, summary_dir=config['trainer']['summary_dir'])
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
      if config['inference']['event_display']:
        colors = colorscale=px.colors.qualitative.G10
	#true
        c = data['c'][mask, :-1]
        c = np.array(c)
        showers=[]
        difuse=[]
        mic=[]
        hip=[]
        mu=[]
        pi=[]
        for idx in range(y_true.shape[0]):
         if y_true[idx].item() == 0:
           showers.append(c[idx])
         if y_true[idx].item() == 1:
           diffuse.append(c[idx])
         if y_true[idx].item() == 3:
           mic.append(c[idx])
         if y_true[idx].item() == 2:
           hip.append(c[idx])
         if y_true[idx].item() == 4:
           mu.append(c[idx])
         if y_true[idx].item() == 5:
            pi.append(c[idx])
        showers = torch.FloatTensor(showers)
        difuse  = torch.FloatTensor(difuse)
        mu = torch.FloatTensor(mu)
        pi = torch.FloatTensor(pi)
        mic = torch.FloatTensor(mic)
        hip = torch.FloatTensor(hip)

        df1 = pd.DataFrame(showers, columns=['x', 'y', 'z'])
        df2 = pd.DataFrame(diffuse, columns=['x', 'y', 'z'])
        df3 = pd.DataFrame(mic, columns=['x', 'y', 'z'])
        df4 = pd.DataFrame(hip, columns=['x', 'y', 'z'])
        df5 = pd.DataFrame(mu, columns=['x', 'y', 'z'])
        df6 = pd.DataFrame(pi, columns=['x', 'y', 'z'])

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=df1.x, y=df1.y, z=df1.z, mode = 'markers',
                           marker=dict(color=colors[4], size=1), name='showers'))

        fig.add_trace(go.Scatter3d(x=df2.x, y=df2.y, z=df2.z, mode = 'markers',
                           marker=dict(color=colors[1], size=1), name='diffuse'))

        fig.add_trace(go.Scatter3d(x=df3.x, y=df3.y, z=df3.z, mode = 'markers',
                           marker=dict(color=colors[2], size=1), name='michel'))

        fig.add_trace(go.Scatter3d(x=df4.x, y=df4.y, z=df4.z, mode = 'markers',
                           marker=dict(color=colors[3], size=1), name='hip'))

        fig.add_trace(go.Scatter3d(x=df5.x, y=df5.y, z=df5.z, mode = 'markers',
                           marker=dict(color=colors[0], size=1), name='mu'))

        fig.add_trace(go.Scatter3d(x=df6.x, y=df6.y, z=df6.z, mode = 'markers',
                           marker=dict(color=colors[5], size=1), name='pi'))


        fig.update_traces(showlegend=True)
        plotly.offline.plot(fig, filename =f'plots/evd/evd_{i}_{j}_true.html', auto_open=False)
        del showers, difuse, mic, hip, mu, pi 
        del df1, df2, df3, df4, df5, df6
        del fig
	#prediction
        showers =[]
        difuse  =[]
        mic =[]
        hip =[]
        mu = []
        pi = []
        for idx in range(y_pred.shape[0]):
          if y_pred[idx].item() == 0:
            showers.append(c[idx])
          if y_pred[idx].item() == 1:
            diffuse.append(c[idx])
          if y_pred[idx].item() == 3:
            mic.append(c[idx])
          if y_pred[idx].item() == 2:
            hip.append(c[idx])
          if y_pred[idx].item() == 4:
            mu.append(c[idx])
          if y_pred[idx].item() == 5:
            pi.append(c[idx])
        showers = torch.FloatTensor(showers)
        difuse  = torch.FloatTensor(difuse)
        mu = torch.FloatTensor(mu)
        pi = torch.FloatTensor(pi)
        mic = torch.FloatTensor(mic)
        hip = torch.FloatTensor(hip)

        df1 = pd.DataFrame(showers, columns=['x', 'y', 'z'])
        df2 = pd.DataFrame(diffuse, columns=['x', 'y', 'z'])
        df3 = pd.DataFrame(mic, columns=['x', 'y', 'z'])
        df4 = pd.DataFrame(hip, columns=['x', 'y', 'z'])
        df5 = pd.DataFrame(mu, columns=['x', 'y', 'z'])
        df6 = pd.DataFrame(pi, columns=['x', 'y', 'z'])

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(x=df1.x, y=df1.y, z=df1.z, mode = 'markers',
                           marker=dict(color=colors[4], size=1), name='showers'))

        fig.add_trace(go.Scatter3d(x=df2.x, y=df2.y, z=df2.z, mode = 'markers',
                           marker=dict(color=colors[1], size=1), name='diffuse'))

        fig.add_trace(go.Scatter3d(x=df3.x, y=df3.y, z=df3.z, mode = 'markers',
                           marker=dict(color=colors[2], size=1), name='michel'))

        fig.add_trace(go.Scatter3d(x=df4.x, y=df4.y, z=df4.z, mode = 'markers',
                           marker=dict(color=colors[3], size=1), name='hip'))

        fig.add_trace(go.Scatter3d(x=df5.x, y=df5.y, z=df5.z, mode = 'markers',
                           marker=dict(color=colors[0], size=1), name='mu'))

        fig.add_trace(go.Scatter3d(x=df6.x, y=df6.y, z=df6.z, mode = 'markers',
                           marker=dict(color=colors[5], size=1), name='pi'))


        fig.update_traces(showlegend=True)
        plotly.offline.plot(fig, filename =f'plots/evd/evd_{i}_{j}_pred.html', auto_open=False)
        del showers, difuse, mic, hip, mu, pi
        del df1, df2, df3, df4, df5, df6
        del fig


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

