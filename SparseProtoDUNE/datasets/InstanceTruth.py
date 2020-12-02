'''Code to produce instance segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle
import math
def get_track_list(pdg, trackId, ener, coo): 
  coo = np.array(coo)
  points = []
  tracks = np.ones([len(pdg)], dtype=np.float32)*(-1)
  energy   = np.zeros([len(pdg)], dtype=np.float32)

## Get a unique list of track ids.
  for i in range(len(pdg)):
    if len(pdg[i]) > 0:
      index = ener[i].index(max(ener[i]))
      if abs(pdg[i][index]) != 11: 
        tracks[i] = trackId[i][index] 
        #energy[i] = ener[i][index] 
  unique_tracks, counts = np.unique(tracks,return_counts=True)

  return tracks, unique_tracks, counts    


def get_Mahalanobis_distance(x,mu,sigmainv):
    x = np.array(x)
    M = (x - mu).dot(sigmainv).dot((x-mu).transpose())
    d= math.sqrt(M)
    return M

def get_InstanceTruth(c,y,voxid):
  Sigmainv = []
  centroids = []
  offsets = np.zeros(c.shape[0])*(-1)
  e = np.zeros([c.shape[0],1])
  for k in range (e.shape[0]):
    e[k] = y[k,y.argmax(dim=1)[k]].item() ##  highest energy contribution from each voxel
  u  = np.unique(voxid)
  for i in range(len(u)):
    mask = (voxid== u[i])
      #cm = [0,0,0] ## background. Just to keep equal dimensions in vectors
      #offsets[mask] = -1
    if i!=0:
      #print('Debug', u[i], e[mask].sum())
      ew = e[mask]/e[mask].sum() #energy weights 
      cm = (np.array(c[mask])*ew).sum(axis=0)
      S = ((ew*(np.array(c[mask]) - cm)).transpose()).dot((np.array(c[mask]) - cm))
      if  S[0,0] ==0. or S[1,1]==0. or S[2,2]==0.:
        S += np.identity(3)
      Sinv = np.linalg.inv(S)
      #Sigmainv.append(np.triu(Sinv))
      Sigmainv.append(Sinv)
      centroids.append(cm)
  offsets = np.zeros([c.shape[0],len(centroids)])
  for idx in range(c.shape[0]):
    for jdx in range(len(centroids)):
      D = get_Mahalanobis_distance(c[idx],centroids[jdx],Sigmainv[jdx])
      offsets[idx,jdx] = D
  return centroids, offsets, Sigmainv

