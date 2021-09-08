#!/usr/bin/env python

import sys
sys.path.append("/scratch") # This line is equivalent to doing source scripts/source_me.sh in a bash terminal
import argparse, yaml
from glob import glob
from random import shuffle

parser = argparse.ArgumentParser("create_shuffled_dataset.py")
parser.add_argument("config", nargs="?", default="/scratch/SparseNOvA/config/nova_sparse_fishnet.yaml")
with open(parser.parse_args().config) as f:
  config = yaml.load(f, Loader=yaml.FullLoader)

topdir = config["data"]["topdir"]
nonswap = glob(topdir+"/nonswap_nu/*.pt")
tauswap = glob(topdir+"/tauswap_nu/*.pt")
fluxswap = glob(topdir+"/fluxswap_nu/*.pt")
nonswap_cosmics = glob(topdir+"/nonswap_cosmic/*.pt")
tauswap_cosmics = glob(topdir+"/tauswap_cosmic/*.pt")
fluxswap_cosmics = glob(topdir+"/fluxswap_cosmic/*.pt")

all_nus = nonswap + tauswap + fluxswap
all_cosmics = nonswap_cosmics + tauswap_cosmics + fluxswap_cosmics

max_cosmics = int(0.1 * len(all_nus))
if len(all_cosmics) > max_cosmics: all_cosmics = all_cosmics[0:max_cosmics]

full_dataset = all_nus + all_cosmics
shuffle(full_dataset)

val_frac = config["data"]["val_fraction"]
test_frac = config["data"]["test_fraction"]

boundary = 1 - (val_frac + test_frac)
split1 = int(boundary * len(full_dataset))
boundary += val_frac
split2 = int(boundary * len(full_dataset))

train_dataset = full_dataset[:split1]
val_dataset = full_dataset[split1:split2]
test_dataset = full_dataset[split2:]

print(f"{len(full_dataset)} files: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")

def link_files(files, topdir, filetype):
  import tqdm, os, os.path as osp
  from uuid import uuid4 as uuid
  print(f"linking {filetype} files...")
  for f in tqdm.tqdm(files):
    outpath = f"{topdir}/{filetype}/{uuid()}.pt"
    os.symlink(f, outpath)

link_files(train_dataset, topdir, "training")
link_files(val_dataset, topdir, "validation")
link_files(test_dataset, topdir, "testing")

