import os, sys, random
import pandas as pd, h5py, numpy as np, torch
from pandana.core import *
from Utils.index import KL, index
from uuid import uuid4 as uuidgen

# DQ
kVeto = Cut(lambda tables: (tables['rec.sel.veto']['keep'] == 1).groupby(level=KL).first())
kVtx  = Cut(lambda tables: (tables['rec.vtx.elastic']['IsValid'] == 1).groupby(level=KL).first())
kPng  = Cut(lambda tables: (tables['rec.vtx.elastic.fuzzyk']['npng'] > 0).groupby(level=KL).first())
kFEB  = Cut(lambda tables: (tables['rec.sel.nuecosrej']['hitsperplane'] < 8).groupby(level=KL).first())

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

def kNueOrNumu(tables):
  pdg = tables['rec.mc.nu']['pdg']
  cc = tables['rec.mc.nu']['iscc']
  return (((pdg==12) | (pdg==14) | (pdg==-12) | (pdg==-14)) & (cc==1)).groupby(level=KL).first()
kNueOrNumu = Cut(kNueOrNumu)

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
  if 0 <= row.interaction < 4:  return 0
  elif 4 <= row.interaction < 8:  return 1
  elif 8 <= row.interaction < 12: return 2
  elif row.interaction == 13:   return 3
  else:               return 4

if __name__ == '__main__':
  # Miniprod 5 h5s
  indir = sys.argv[1]
  outdir = sys.argv[2]
  print('Change files in '+indir+' to training files in '+outdir)
  files = [f for f in os.listdir(indir) if 'h5caf.h5' in f]
  files = random.sample(files, len(files))
  print('There are '+str(len(files))+' files.')

  # Full selection
  kCut = kVeto & kNueOrNumu & kContain & kVtx & kPng & kFEB

  # One file at a time to avoid problems with loading a bunch of pixel maps in memory
  for i,f in enumerate(files):

    # Definte the output name and don't recreate it
    outname = '{0}_TrainData{1}'.format(f[:-9], f[-9:])
    if os.path.exists(os.path.join(outdir,outname)):
      continue
    
    # Make a loader and the two spectra to fill
    tables = Loader([os.path.join(indir,f)], idcol='evt.seq', main_table_name='spill', indices=index)
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
    if specLab.entries()==0 or specMap.entries()==0:
      print(str(i)+': File '+f+' is empty.')
      continue
    
    # Concat the dataframes to line up label and map
    # join='inner' ensures there is both a label and a map for the slice
    df = pd.concat([specLabel.df(), specMap.df(), specSign.df(), specEnergy.df(), specObj.df(), specLab.df(), specFirstCellX.df(), specFirstCellY.df(), specFirstPlane.df()], axis=1, join='inner').reset_index()
    
    # Save in an h5
#    hf = h5py.File(os.path.join(outdir,outname),'w')
#    hf.create_dataset('run',     data=df['run'],     compression='gzip')
#    hf.create_dataset('subrun',  data=df['subrun'],    compression='gzip')
#    hf.create_dataset('cycle',   data=df['cycle'],     compression='gzip')
#    hf.create_dataset('event',   data=df['evt'],     compression='gzip')
#    hf.create_dataset('slice',   data=df['subevt'],    compression='gzip')
#    hf.create_dataset('label',   data=df['interaction'], compression='gzip')
#    hf.create_dataset('PDG',     data=df['pdg'],     compression='gzip')
#    hf.create_dataset('E',     data=df['nuenergy'],  compression='gzip')
#    hf.create_dataset('cvnmap',  data=np.stack(df['cvnmap']), compression='gzip')
#    hf.create_dataset('cvnobjmap', data=np.stack(df['cvnobjmap']), compression='gzip')
#    hf.create_dataset('cvnlabmap', data=np.stack(df['cvnlabmap']), compression='gzip')
#    hf.create_dataset('firstcellx',   data=np.stack(df['firstcellx']), compression='gzip')
#    hf.create_dataset('firstcelly',   data=np.stack(df['firstcelly']), compression='gzip')
#    hf.create_dataset('firstplane',   data=np.stack(df['firstplane']), compression='gzip')
#    hf.close()

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
      zoffset = np.floor(row.firstplane/2)
      xoffset = row.firstcellx
      yoffset = row.firstcelly

      # get pixel coordinates
      xcoord = np.stack(xmask, axis=0) + np.array([zoffset, xoffset])[:,None]
      ycoord = np.stack(ymask, axis=0) + np.array([zoffset, yoffset])[:,None]

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

      torch.save(data, '{}/{}.pt'.format(outdir, uuidgen()))

    df.apply(process_evt, axis=1)

