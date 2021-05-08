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

full_dataset = datasets.get_dataset(**config['data'])

fulllen = len(full_dataset)
tv_num = math.ceil(fulllen*config['data']['t_v_split'])
splits = np.cumsum([fulllen-tv_num,0,tv_num])

train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))

# parameters = [sherpa.Continuous('learning_rate', [1e-5, 1e-1]), sherpa.Continuous('weight_decay', [0.01, 0.1]), sherpa.Discrete('unet_depth', [2, 6])]

trainer = Trainer(**config['trainer'])

# alg = sherpa.algorithms.GPyOpt(max_num_trials=50)
# study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=True, dashboard_port=8000)

collate_train = getattr(utils, config['model']['collate_train'])
collate_valid = getattr(utils, config['model']['collate_valid'])

train_loader = DataLoader(train_dataset, collate_fn=collate_train, **config['data_loader'], shuffle=True)
valid_loader = DataLoader(valid_dataset, collate_fn=collate_valid, **config['data_loader'], shuffle=False)

trainer.build_model(**config['model'])

# Once all the setup is done, all that's left is to run training and save some summary statistics to file.
train_summary = trainer.train(train_loader, valid_data_loader=valid_loader, **config['trainer'])
print(train_summary)
torch.save(train_summary, 'summary_test.pt')

