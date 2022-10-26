#!/usr/bin/env python
import os, os.path as osp, sys, argparse, random
import pandas as pd, h5py, numpy as np, torch
from pandana.core import *
from Utils.index import KL, index
from uuid import uuid4 as uuidgen
import matplotlib.pyplot as plt

# DQ
kVeto = Cut(lambda tables: (tables['rec.sel.veto']['keep'] == 1).groupby(level=KL).first())
kVtx  = Cut(lambda tables: (tables['rec.vtx.elastic']['IsValid'] == 1).groupby(level=KL).first())
kPng  = Cut(lambda tables: (tables['rec.vtx.elastic.fuzzyk']['npng'] > 0).groupby(level=KL).first())
kFEB  = Cut(lambda tables: (tables['rec.sel.nuecosrej']['hitsperplane'] < 8).groupby(level=KL).first())
kNoCut = Cut(lambda tables: tables['rec.slc']['nhit'] > 0)

# Containment
def kContain(tables):
  df = tables['rec.sel.nuecosrej']
  return (\
    (df['distallpngtop'] > 30) & \
    (df['distallpngbottom'] > 30) & \
    (df['distallpngeast'] > 30) & \
    (df['distallpngwest'] > 30) & \
    (df['distallpngfront'] > 30) & \
    (df['distallpngback'] > 30).groupby(level=KL).first())
kContain = Cut(kContain)

def kNCAndCC(tables):
  pdg = tables['rec.mc.nu']['pdg']
  cc = tables['rec.mc.nu']['iscc']
  return ((abs(pdg)==12) | (abs(pdg)==14) | (abs(pdg)==16)).groupby(level=KL).first()
kNCAndCC = Cut(kNCAndCC)

def kNoTau(tables):
  pdg = tables['rec.mc.nu']['pdg']
  return ((abs(pdg)==12) | (abs(pdg)==14)).groupby(level=KL).first()
kNoTau = Cut(kNoTau)

def kCC(tables):
  pdg = tables['rec.mc.nu']['pdg']
  cc = tables['rec.mc.nu']['iscc']
  return (((pdg==12) | (pdg==14) | (pdg==16) | (pdg==-12) | (pdg==-14) | (pdg==-16)) & (cc==1)).groupby(level=KL).first()
kCC = Cut(kCC)

def kCosmic(tables):
  cos = tables['rec.training.trainingdata']['interaction']
  return ((cos==15).groupby(level=KL).first())
kCosmic = Cut(kCosmic)

def kSign(tables):
  return tables['rec.mc.nu']['pdg'].groupby(level=KL).first()
kSign = Var(kSign)

# Labels and maps for CVN training
def kLabel(tables):
  return tables['rec.training.trainingdata']['interaction'].groupby(level=KL).first()
kLabel = Var(kLabel)

def kEnergy(tables):
  return tables['rec.training.trainingdata']['nuenergy'].groupby(level=KL).first()
kEnergy = Var(kEnergy)

def kMap(tables):
  return tables['rec.training.cvnmaps']['cvnmap'].groupby(level=KL).first()
kMap = Var(kMap)

def kObj(tables):
  return tables['rec.training.cvnmaps']['cvnobjmap'].groupby(level=KL).first()
kObj = Var(kObj)

def kLab(tables):
  return tables['rec.training.cvnmaps']['cvnlabmap'].groupby(level=KL).first()
kLab = Var(kLab)

def kFirstCellX(tables):
  return tables['rec.training.cvnmaps']['firstcellx'].groupby(level=KL).first()
kFirstCellX = Var(kFirstCellX)

def kFirstCellY(tables):
  return tables['rec.training.cvnmaps']['firstcelly'].groupby(level=KL).first()
kFirstCellY = Var(kFirstCellY)

def kFirstPlane(tables):
  return tables['rec.training.cvnmaps']['firstplane'].groupby(level=KL).first()
kFirstPlane = Var(kFirstPlane)

def get_alias(row):
  # [0 == nu_mu, 1 == nu_e, 2 == nu_tau, 3 == NC, 4 == others]
  if 0 <= row.interaction < 4:  return 0
  elif 4 <= row.interaction < 8:  return 1
  elif 8 <= row.interaction < 12:  return 2
  elif row.interaction == 13:   return 3
  else:               return 4

if __name__ == '__main__':
  # Miniprod 5 h5s
  args = argparse.ArgumentParser(description="Preprocess NOvA HDF5s into ML training inputs")
  args.add_argument("-i", "--indir", type=str, required=True,
                    help="directory containing input HDF5 files")
  args.add_argument("-o", "--outdir", type=str, required=True,
                    help="directory to write output files to")
  args.add_argument("-t", "--type", type=str, required=True, choices=["nu", "notau", "cosmic"],
                    help="which selection to apply")
  opts = args.parse_args()
  print('Change files in '+opts.indir+' to training files in '+opts.outdir)
  files = [f for f in os.listdir(opts.indir) if 'h5caf.h5' in f]
  files = random.sample(files, len(files))
  print('There are '+str(len(files))+' files.')

  # Full selection
  kCut = kVeto & kContain & kVtx & kPng & kFEB
  if opts.type == "nu": kCut = kCut & kNCAndCC
  if opts.type == "notau": kCut = kCut & kNoTau
  else: kCut = kCut & kCosmic

  # One file at a time to avoid problems with loading a bunch of pixel maps in memory
  from tqdm import tqdm
  for i, f in enumerate(tqdm(files)):

    # Make a loader and the two spectra to fill
    tables = Loader([osp.join(opts.indir,f)], idcol='evt.seq', main_table_name='spill', indices=index)
    specLabel  = Spectrum(tables, kCut, kLabel)
    specMap  = Spectrum(tables, kCut, kMap)
    specSign   = Spectrum(tables, kCut, kSign)
    specEnergy = Spectrum(tables, kCut, kEnergy)
    specObj  = Spectrum(tables, kCut, kObj)
    specLab  = Spectrum(tables, kCut, kLab)
    specFirstCellX = Spectrum(tables, kCut, kFirstCellX)
    specFirstCellY = Spectrum(tables, kCut, kFirstCellY)
    specFirstPlane = Spectrum(tables, kCut, kFirstPlane)
    
    # GO GO GO
    tables.Go()
    
    # Don't save an empty file
    if specLab.entries() == 0 or specMap.entries() == 0: continue
    
    df = pd.concat([specLabel.df(), specMap.df(), specObj.df(), specLab.df(), specFirstCellX.df(), specFirstCellY.df(), specFirstPlane.df()], axis=1, join='inner').reset_index()
    df['label'] = df.apply(get_alias, axis=1)

    def process_evt(row):
      # reshape the cvnmap and label tensors to correct shape
      xview, yview = row.cvnmap.reshape(2, 100, 80)
      xobjmap, yobjmap = row.cvnobjmap.reshape(2, 100, 80)
      xlabmap, ylabmap = row.cvnlabmap.reshape(2, 100, 80)

      # get the indices of the nonzero pixels with mask
      xmask = xview.nonzero()
      ymask = yview.nonzero()

      # get the truth
      truth = row.label

      # get offset coordinates
#       zoffset = np.floor(row.firstplane/2)
#       xoffset = row.firstcellx
#       yoffset = row.firstcelly

      # get pixel coordinates
#       xcoord = np.stack(xmask, axis=0) + np.array([zoffset, xoffset])[:,None]
#       ycoord = np.stack(ymask, axis=0) + np.array([zoffset, yoffset])[:,None]
      xcoord = np.stack(xmask, axis=0)
      ycoord = np.stack(ymask, axis=0)

      # use the masks to get data dictionary
      data = { 'xfeats': torch.tensor(xview[xmask]).unsqueeze(dim=-1).float(),
               'xcoords': torch.tensor(xcoord).T.contiguous().int(),
               'xsegtruth': torch.tensor(xlabmap[xmask]).long(),
               'xinstruth': torch.tensor(xobjmap[xmask]).long(),
               'yfeats': torch.tensor(yview[ymask]).unsqueeze(dim=-1).float(),
               'ycoords': torch.tensor(ycoord).T.contiguous().int(),
               'ysegtruth': torch.tensor(ylabmap[ymask]).long(),
               'yinstruth': torch.tensor(yobjmap[ymask]).long(),
               'evttruth': torch.tensor(truth).long() }

      fname = "b{}_c{}_r{}_sr{}_e{}_se{}.pt".format(row.batch, row.cycle, row.run,
                                                    row.subrun, row.evt, row.subevt)
      torch.save(data, osp.join(opts.outdir, fname))

    df.apply(process_evt, axis=1)

