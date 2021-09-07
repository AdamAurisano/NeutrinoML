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
parser.add_argument('config', nargs='?', default='/scratch/SparseNOvA/config/nova_sparse_fishnet.yaml')
with open(parser.parse_args().config) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

# Here we load the dataset and the trainer, which is responsible for building the model and overseeing training. There's a block of code which is responsible for slicing the full dataset up into a training dataset and a validation dataset where jitter is applied to training dataset only.

train_dataset = datasets.get_dataset(subdir="training", apply_jitter=True, normalize_coord=True, **config['data'])
valid_dataset = datasets.get_dataset(subdir="validation", apply_jitter=False, normalize_coord=True, **config['data'])

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

