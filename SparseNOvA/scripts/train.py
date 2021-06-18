#!/usr/bin/env python

# First we need to correctly set the python environment. This is done by adding the top directory of the repository to the Python path. Once that's done, we can import various packages from inside the repository.
import sys, os.path as osp, yaml, argparse, logging, math, numpy as np, torch, sherpa, logging
sys.path.append('/scratch') # This line is equivalent to doing source scripts/source_me.sh in a bash terminal
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from Core.trainers import Trainer
from glob import glob
from Core import utils
from SparseNOvA import datasets
from Core import models

# Most of the training options are set in a configuration YAML file. We're going to load this config, and then the options inside will be passed to the relevent piece of the training framework.
parser = argparse.ArgumentParser('train.py')
parser.add_argument('config', nargs='?', default='/scratch/SparseNOvA/config/sparse_nova_mobilenet.yaml')
with open(parser.parse_args().config) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

# Here we load the dataset and the trainer, which is responsible for building the model and overseeing training. There's a block of code which is responsible for slicing the full dataset up into a training dataset and a validation dataset where jitter is applied to training dataset only.
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

train_files = all_nus[0:splits_nu[1]] + all_cosmics[0:splits_cos[1]]
train_files.sort(key = lambda x: osp.basename(x))  
train_dataset = datasets.get_dataset(filelist=train_files, apply_jitter=True, normalize_coord=True, **config['data'])

valid_files = all_nus[splits_nu[1]:splits_nu[2]] + all_cosmics[splits_cos[1]:splits_cos[2]]
valid_files.sort(key = lambda x: osp.basename(x))
valid_dataset = datasets.get_dataset(filelist=valid_files, apply_jitter=False, normalize_coord=True, **config['data'])

# parameters = [sherpa.Continuous('learning_rate', [1e-5, 1e-1]), sherpa.Continuous('weight_decay', [0.01, 0.1]), sherpa.Discrete('unet_depth', [2, 6])]

trainer = Trainer(**config['trainer'])

# alg = sherpa.algorithms.GPyOpt(max_num_trials=50)
# study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True, dashboard_port=8000)

collate = getattr(utils, config['model']['collate'])

train_loader = DataLoader(train_dataset, collate_fn=collate, **config['data_loader'], shuffle=True)
valid_loader = DataLoader(valid_dataset, collate_fn=collate, **config['data_loader'], shuffle=False)

trainer.build_model(**config['model'])

# Once all the setup is done, all that's left is to run training and save some summary statistics to file.
train_summary = trainer.train(train_loader, valid_data_loader=valid_loader, **config['trainer'])
print(train_summary)
torch.save(train_summary, 'summary_test.pt')

