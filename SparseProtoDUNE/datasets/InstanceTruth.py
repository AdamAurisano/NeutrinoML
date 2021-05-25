'''Code to produce instance segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle
import math

def get_track_list(pdg, trackId, ener): 
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

def gauss_pdf(c,sigmainv,mu,dim=3):
  prob = []
  sigma = np.linalg.inv(sigmainv)
  den = (2*np.pi)**dim*np.linalg.det(sigma)
  N = 1/np.sqrt(den)
  mu = np.array(mu)
  for k in c:
    k = np.array(k)
    p = (k-mu).dot(sigmainv).dot((k-mu).transpose())
    f = N*np.exp(p*-0.5)
    prob.append(f)
  return prob

def get_InstanceTruth(c,voxid,width):
  sigma = np.identity(3)*width
  sigmainv = np.linalg.inv(sigma)
  medoids = []
  offset = np.ones((c.shape[0],c.shape[1]))*(-1)
  chtm = np.zeros(c.shape[0])
  u = np.unique(voxid)
  for p in np.unique(u[1:]): # start from 1 to avoid background voxels 
    mask = voxid ==p 
    ci = c[mask]
    a = np.zeros(ci.shape[0])
    l = tuple(np.linalg.norm((k -ci),axis=1).sum() for k in ci)
    ind = l.index((min(l)))
    medoids.append(np.array(ci[ind]))
    Off = tuple(np.array(k-ci[ind]) for k in ci)
    chtm[mask]= gauss_pdf(ci,sigmainv,ci[ind])
    #a[ind] = 1
    #chtm[mask] = a 
    offset[mask] = Off
  chtm = chtm.reshape([chtm.shape[0],1])
  chtm = chtm/(chtm.max().item())
  return np.array(medoids), chtm, offset
