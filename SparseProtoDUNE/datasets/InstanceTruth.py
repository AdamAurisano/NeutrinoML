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
  unique_tracks, counts = np.unique(tracks,return_counts=True)

  return tracks, unique_tracks, counts    


def get_Mahalanobis_distance(x,mu,covm):
  x = np.array(x)
  sigmainv = np.linalg.inv(covm)
  M = (x - mu).dot(sigmainv).dot((x-mu).transpose())
  d= math.sqrt(M)
  return M

def get_InstanceTruth(c,y,voxid):
  CovM = []
  centroids = []
  #offsets = np.zeros(c.shape[0])*(-1)
  e = np.zeros([c.shape[0],1])
  for k in range (e.shape[0]):
    e[k] = y[k,y.argmax(dim=1)[k]].item() ##  highest energy contribution from each voxel
  u  = np.unique(voxid)
  for i in range(len(u)):
    mask = (voxid== u[i])
    if i!=0:
      #ew = e[mask]/e[mask].sum() #energy weights 
      #cm = (np.array(c[mask])*ew).sum(axis=0)
      #S = ((ew*(np.array(c[mask]) - cm)).transpose()).dot((np.array(c[mask]) - cm))
      cm = np.array(c[mask]).sum(axis=0)/c[mask].shape[0]
      S = (((np.array(c[mask]) - cm)).transpose()).dot((np.array(c[mask]) - cm))/(c[mask].shape[0]-1)
      for k in range(3):
        if S[k,k] == 0: S[k,k]=1.e-4
      #Sigmainv.append(np.triu(Sinv))
      Sinv = np.linalg.inv(S)
      CovM.append(S)
      centroids.append(cm)
  offsets = np.zeros([c.shape[0],len(centroids)])
  for idx in range(c.shape[0]):
    for jdx in range(len(centroids)):
      D = get_Mahalanobis_distance(c[idx],centroids[jdx],CovM[jdx])
      offsets[idx,jdx] = D
  return centroids, offsets, CovM

