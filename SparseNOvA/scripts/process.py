#!/usr/bin/env python
# coding: utf-8

# ### Imports
# 
# We need to pull in a few packages first

import torch, h5py, numpy as np, glob, tqdm, sys, multiprocessing as mp
from uuid import uuid4 as uuidgen
sys.path.append('/scratch') # Set up local python environment
from Core import utils
from SparseNOvA import datasets

def match_truth(file, im_idx):
    '''Returns list of index locations of trainingdata for a given slicemap'''
    keys = ['run','subrun','cycle','evt','subevt']
    tag = { key: file['rec.training.slicemaps'][key][im_idx,0] for key in keys }
    mask = None
    for key in keys:
        m = (file['rec.training.trainingdata'][key][:,0] == tag[key])
        mask = m if mask is None else mask & m
    true_idx = mask.nonzero()[0]
    if len(true_idx) > 1:
        raise Exception(f'{len(true_idx)} truths found for slicemap {im_idx}.')
    return true_idx

def get_alias(flav):
    '''Function that alias interaction enum to the following list'''
    # [0 == nu_mu, 1 == nu_e, 2 == nu_tau, 3 == NC, 4 == others]
    if 0 <= flav < 4:    return 0
    elif 4 <= flav < 8:  return 1
    elif 8 <= flav < 12: return 2
    elif flav == 13:     return 3
    else:                return 4

def process_file(filename):
    
    file = h5py.File(filename, 'r')
    mask = np.nonzero(file['rec.mc']['nnu'][:,0]==1)[0]

    # Loop over each neutrino image to look for the associated truth
    for i, nu in enumerate(mask):
        true_idx = match_truth(file, nu)
        if len(true_idx) == 0: continue
        image = file['rec.training.slicemaps']['slicemap'][nu]
        xview, yview = image.reshape(2, 448, 384)[:]
        truth = get_alias(file['rec.training.trainingdata']['interaction'][true_idx,0][0])
        xsparse = torch.tensor(xview).float().to_sparse()
        ysparse = torch.tensor(yview).float().to_sparse()
        data = { 'xfeats': xsparse._values().unsqueeze(dim=-1),
                 'xcoords': xsparse._indices().T.int(),
                 'yfeats': ysparse._values().unsqueeze(dim=-1),
                 'ycoords': ysparse._indices().T.int(),
                 'truth': torch.tensor(truth).long() }
        torch.save(data, f'/data/mp5/processed/{uuidgen()}.pt')


# ### Preprocessing
# 
# Pull the interesting events out of the HDF5 files, preprocess them, and write them as individual PyTorch files instead.

nonswap = sorted(glob.glob('/data/mp5/nonswap/*.h5'))
fluxswap = sorted(glob.glob('/data/mp5/fluxswap/*.h5'))
tauswap = sorted(glob.glob('/data/mp5/tauswap/*.h5'))
files = nonswap + fluxswap + tauswap

with mp.Pool(processes=50) as pool: pool.map(process_file, files)


# ### Filtering
# 
# Remove all images with no hits in either view

files = glob.glob('/data/mp5/processed/*.pt')
for filename in files:
    data = torch.load(filename)
    if data['xcoords'].shape[0] == 0 or data['ycoords'].shape[0] == 0:
        print(f'{filename} is bad, removing.')
        os.remove(filename)

