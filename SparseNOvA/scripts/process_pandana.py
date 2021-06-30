import os
import sys
import random
import numpy as np
import torch
from uuid import uuid4 as uuidgen
import pandana
from pandana import *
# DQ
kVeto = Cut(lambda tables: tables['rec.sel.veto']['keep'] == 1)
kVtx  = Cut(lambda tables: tables['rec.vtx.elastic']['IsValid'] == 1)
kPng  = Cut(lambda tables: tables['rec.vtx.elastic.fuzzyk']['npng'] > 0)
kFEB  = Cut(lambda tables: tables['rec.sel.nuecosrej']['hitsperplane'] < 8)
# Containment
def kContain(tables):
    df = tables['rec.sel.nuecosrej']
    return \
        (df['distallpngtop'] > 30) & \
        (df['distallpngbottom'] > 30) & \
        (df['distallpngeast'] > 30) & \
        (df['distallpngwest'] > 30) & \
        (df['distallpngfront'] > 30) & \
        (df['distallpngback'] > 30)
kContain = Cut(kContain)
def kNueOrNumu(tables):
    pdg = tables['rec.mc.nu']['pdg']
    cc = tables['rec.mc.nu']['iscc']
    return ((pdg==12) | (pdg==14) | (pdg==-12) | (pdg==-14)) & (cc==1)
kNueOrNumu = Cut(kNueOrNumu)
def kSign(tables):
    return tables['rec.mc.nu']['pdg']
kSign = Var(kSign)
# Labels and maps for CVN training
def kLabel(tables):
    return tables['rec.training.trainingdata']['interaction']
kLabel = Var(kLabel)
def kEnergy(tables):
    return tables['rec.training.trainingdata']['nuenergy']
kEnergy = Var(kEnergy)
def kMap(tables):
    return tables['rec.training.cvnmaps']['cvnmap']
kMap = Var(kMap)
def kObj(tables):
    return tables['rec.training.cvnmaps']['cvnobjmap']
kObj = Var(kObj)
def kLab(tables):
    return tables['rec.training.cvnmaps']['cvnlabmap']
kLab = Var(kLab)
def get_alias(row):
    if 0 <= row.interaction < 4:    return 0
    elif 4 <= row.interaction < 8:  return 1
    elif 8 <= row.interaction < 12: return 2
    elif row.interaction == 13:     return 3
    else:                           return 4

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
        tables = loader([os.path.join(indir,f)])
        specLabel  = spectrum(tables, kCut, kLabel)
        specMap    = spectrum(tables, kCut, kMap)
        specSign   = spectrum(tables, kCut, kSign)
        specEnergy = spectrum(tables, kCut, kEnergy)
        specObj    = spectrum(tables, kCut, kObj)
        specLab    = spectrum(tables, kCut, kLab)
        # GO GO GO
        tables.Go()
        # Don't save an empty file
        if specLab.entries()==0 or specMap.entries()==0:
            print(str(i)+': File '+f+' is empty.')
            continue
        # Concat the dataframes to line up label and map
        # join='inner' ensures there is both a label and a map for the slice
        df = pd.concat([specLabel.df(), specMap.df(), specSign.df(), specEnergy.df(), specObj.df(), specLab.df()], axis=1, join='inner').reset_index()
        # Save in an h5
        hf = h5py.File(os.path.join(outdir,outname),'w')
        hf.create_dataset('run',       data=df['run'],         compression='gzip')
        hf.create_dataset('subrun',    data=df['subrun'],      compression='gzip')
        hf.create_dataset('cycle',     data=df['cycle'],       compression='gzip')
        hf.create_dataset('event',     data=df['evt'],         compression='gzip')
        hf.create_dataset('slice',     data=df['subevt'],      compression='gzip')
        hf.create_dataset('label',     data=df['interaction'], compression='gzip')
        hf.create_dataset('PDG',       data=df['pdg'],         compression='gzip')
        hf.create_dataset('E',         data=df['nuenergy'],    compression='gzip')
        hf.create_dataset('cvnmap',    data=np.stack(df['cvnmap']), compression='gzip')
        hf.create_dataset('cvnobjmap',    data=np.stack(df['cvnobjmap']), compression='gzip')
        hf.create_dataset('cvnlabmap',    data=np.stack(df['cvnlabmap']), compression='gzip')
        hf.close()
        df['label'] = df.apply(get_alias, axis=1)
        for j, row in df.iterrows():
            # reshape the cvnmap and label tensors to correct shape
            xview, yview = row.cvnmap.reshape(2, 100, 80)
            xobjmap, yobjmap = row.cvnobjmap.reshape(2, 100, 80)
            xlabmap, ylabmap = row.cvnlabmap.reshape(2, 100, 80)

            # get the indicies of the nonzero pixels with mask
            xmask = xview.nonzero()
            ymask = yview.nonzero()

            # get the truth
            truth = row.label

            # use the masks to get data dictionary
            data = { 'xfeats': torch.tensor(xview[xmask]).float(), 
                     'xcoords': torch.tensor(xmask).int(), 
                     'xsegtruth': torch.tensor(xlabmap[xmask]).long(),
                     'xinstruth': torch.tensor(xobjmap[xmask]).long(), 
                     'yfeats': torch.tensor(yview[ymask]).float(),
                     'ycoords': torch.tensor(ymask).int(),
                     'ysegtruth': torch.tensor(ylabmap[ymask]).long(), 
                     'yinstruth': torch.tensor(yobjmap[ymask]).long(), 
                     'evttruth': torch.tensor(truth).long() }
            
            torch.save(data, '/data/p5/processed/{}.pt'.format(uuidgen()))
