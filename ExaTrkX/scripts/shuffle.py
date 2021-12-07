#!/usr/bin/env python

import sys
sys.path.append("/scratch") # This line is equivalent to doing source scripts/source_me.sh in a bash terminal
import argparse, yaml, glob, random

if len(sys.argv) != 2:
  raise Exception("you must pass a config file as argument!")

with open(sys.argv[1]) as f:
  cfg = yaml.load(f, Loader=yaml.FullLoader)

topdir = cfg["data"]["path"]
files = glob.glob(f"{topdir}/nue/*.pt") + glob.glob(f"{topdir}/numu/*.pt")
random.shuffle(files)

split1 = int(cfg["data"]["test_fraction"] * len(files))
split2 = split1 + int(cfg["data"]["val_fraction"] * len(files))

test_dataset = files[:split1]
val_dataset = files[split1:split2]
train_dataset = files[split2:]

print(f"{len(files)} files: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test")

def link_files(files, topdir, filetype):
  import tqdm, os, uuid
  print(f"linking {filetype} files...")
  for f in tqdm.tqdm(files):
    outpath = os.path.join(topdir, filetype)
    if not os.path.exists(outpath): os.mkdir(outpath)
    link = os.path.join(outpath, f"{uuid.uuid4()}.pt")
    os.symlink(os.path.relpath(f, outpath), link)

link_files(train_dataset, topdir, "train")
link_files(val_dataset, topdir, "valid")
link_files(test_dataset, topdir, "test")

