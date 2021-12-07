#!/usr/bin/env python

# Import modules
def train():
  import sys, os, os.path as osp, argparse, yaml, logging, numpy as np, tqdm
  workdir = osp.dirname(osp.dirname(osp.realpath(__file__)))
  import torch, torch_geometric
  if workdir not in sys.path: sys.path.append(workdir)
  import datasets
  from core.trainers import Trainer
  from torch_geometric.loader import DataLoader

  # Configuration options
  def configure(config):
    """Load input configuration file"""
    with open(config) as f:
      return yaml.load(f, Loader=yaml.FullLoader)

  # Configuration options (overwrite default configuration with your own if you want!)
  parser = argparse.ArgumentParser("train.py")
  add_arg = parser.add_argument
  add_arg("config", nargs="?", default=osp.join(workdir, "config/hit2d.yaml"))
  args = parser.parse_args()
  config = configure(args.config)

  trainer = Trainer(**config["trainer"])

  # Load dataset
  train_dataset = datasets.training_dataset(**config["data"])
  valid_dataset = datasets.validation_dataset(**config["data"])

  weights = train_dataset.load_weights(trainer.device).pow(0.75)
  # weights = torch.ones(config["model"]["output_dim"], device=trainer.device)
  print("class weights:")
  class_names = config["model"]["metric_params"]["Graph"]["class_names"]
  for name, weight in zip(class_names, weights):
    print(f"  {name}: {weight}")

  mean, std = train_dataset.load_norm(trainer.device)
  transform = datasets.FeatureNorm(mean, std)
  config["model"]["loss_params"]["weight"] = weights

  train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], 
    num_workers=config["trainer"]["num_workers"], shuffle=True, pin_memory=True, follow_batch=['x_u', 'x_v', 'x_y'])
  valid_loader = DataLoader(valid_dataset, batch_size=config["trainer"]["batch_size"],
    num_workers=config["trainer"]["num_workers"], shuffle=False, follow_batch=['x_u', 'x_v', 'x_y'])

  # Build model
  trainer.build_model(**config["model"], transform=transform)

  # Train!
  train_summary = trainer.train(train_loader, valid_data_loader=valid_loader, **config["trainer"])
  print(train_summary)

if __name__ == "__main__":
  train()
