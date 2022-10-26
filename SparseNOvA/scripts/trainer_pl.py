#!/usr/bin/env python

import sys
import os
import argparse
import torch
import pytorch_lightning as pl
from SparseNOvA import datasets
from torch.utils.data import DataLoader
from Core.models import LightningFish

import MinkowskiEngine as ME

ME.set_sparse_tensor_operation_mode(ME.SparseTensorOperationMode.SHARE_COORDINATE_MANAGER)

def collate_sparse_minkowski_2stack(batch):
  import MinkowskiEngine as ME
  x_coords   = ME.utils.batched_coordinates([d['xcoords'] for d in batch])
  x_feats    = torch.cat([d['xfeats'] for d in batch])
  x_segtruth = torch.cat([d['xsegtruth'] for d in batch])
  x_instruth = torch.cat([d['xinstruth'] for d in batch])
  y_coords   = ME.utils.batched_coordinates([d['ycoords'] for d in batch])
  y_feats    = torch.cat([d['yfeats'] for d in batch])
  y_segtruth = torch.cat([d['ysegtruth'] for d in batch])
  y_instruth = torch.cat([d['yinstruth'] for d in batch])
  y          = torch.stack([d['evttruth'] for d in batch])
  ret = { 'sparse': [x_feats, x_coords, y_feats, y_coords], 'y': y }
  return ret

def trainer_pl(args):

    # multiprocessing fixes for CPU threads on Wilson
    # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
    print(f'Current Sharing Strategy: {torch.multiprocessing.get_sharing_strategy()}')
    torch.multiprocessing.set_sharing_strategy('file_system')
    print(f'Updated Sharing Strategy: {torch.multiprocessing.get_sharing_strategy()}')

    # Datasets
    train_dataset = datasets.get_dataset(name='SparsePixelMapNOvA',
                                         topdir='/data/p5/processed',
                                         subdir='training',
                                         val_fraction=0.05,
                                         test_fraction=0.05,
                                         apply_jitter=True,
                                         standardize_input=True)
    valid_dataset = datasets.get_dataset(name='SparsePixelMapNOvA',
                                         topdir='/data/p5/processed',
                                         subdir='validation',
                                         val_fraction=0.05,
                                         test_fraction=0.05,
                                         apply_jitter=False,
                                         standardize_input=True)
    
    # Dataloaders
    collate = collate_sparse_minkowski_2stack
    train_loader = DataLoader(train_dataset, 
                              collate_fn=collate, 
                              batch_size=1024,
                              num_workers=5, 
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset, 
                              collate_fn=collate, 
                              batch_size=1024,
                              num_workers=5, 
                              shuffle=False)
    
    
    model = LightningFish(D = 2, 
                                input_feats=3,
                                num_cls = 5,
                                num_down_sample = 3, 
                                num_up_sample = 3, 
                                trans_map = (2, 1, 0, 6, 5, 4),
                                network_planes = [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
                                num_res_blks = [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
                                num_trans_blks = [2, 2, 2, 2, 2, 4],
                                classes=["numu","nue","nutau","nc","cosmic"],
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                # alpha=None,
                                gamma=args.gamma)

    os.makedirs(os.path.join(args.logdir, args.name), exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=args.logdir,
                                          name=args.name)
    profiler = pl.profilers.PyTorchProfiler(filename=f'{logger.log_dir}/profile.txt') \
            if args.profile else None
    if args.devices is None:
        print('No devices specified – training with CPU')

    accelerator = 'cpu' if args.devices is None else 'gpu'
    
    trainer = pl.Trainer(accelerator=accelerator,
                         devices=args.devices,
                         max_epochs=args.max_epochs,
                         logger=logger,
                         profiler=profiler)
    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--name', type=str, required=True,
                        help='Training instance name, for logging purposes')
    parser.add_argument('--devices', nargs='+', type=int, default=None,
                        help='List of devices to train with')
    parser.add_argument('--logdir', type=str, default='/SparseNOvA/logs',
                        help='Output directory to write logs to')
    parser.add_argument('--resume', type=str, default=False,
                        help='Checkpoint file to resume training from')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable PyTorch profiling')
    
    parser = LightningFish.add_train_args(parser)
    args = parser.parse_args()

    trainer_pl(args)
