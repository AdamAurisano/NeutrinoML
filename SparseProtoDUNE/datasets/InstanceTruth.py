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

def gauss_pdf(c,sigmainv,mu,dim=3):
  prob = []
  sigma = np.linalg.inv(sigmainv)
  den = (2*np.pi)**dim*np.linalg.det(sigma)
  N = 1/np.sqrt(den)
  for k in c:
    k = np.array(k)
    p = (k-mu).dot(sigmainv).dot((k-mu).transpose())
    f = N*np.exp(p*-0.5)
    prob.append(f)
  return prob

def get_InstanceTruth(c,y,voxid):
  invCovM = []
  centroids = []
  c_centroids = []
  e = np.zeros([c.shape[0],1])
  for k in range (e.shape[0]):
    e[k] = y[k,y.argmax(dim=1)[k]].item() ##  highest energy contribution from each voxel
  u  = np.unique(voxid) 
  offsets = np.ones(c.shape[0])*(-1)
  prob = np.zeros(c.shape[0])
  fixCovM = np.identity(3)*8
  fixCovMinv = np.linalg.inv(fixCovM)
  for i in range(len(u)):
    mask = (voxid== u[i])
    if i!=0:
      #ew = e[mask]/e[mask].sum() #energy weights 
      #cm = (np.array(c[mask])*ew).sum(axis=0)
      #S = ((ew*(np.array(c[mask]) - cm)).transpose()).dot((np.array(c[mask]) - cm))
      #cm = np.floor(tu)
      #get a centroid in the instance 
      #absolute_difference_function = lambda c[mask] : abs(list_value - given_value)
      #closest_value = min(a_list, key=absolute_difference_function)
      # Get the center of mass
      cm = np.array(c[mask]).sum(axis=0)/c[mask].shape[0]
      S = (((np.array(c[mask]) - cm)).transpose()).dot((np.array(c[mask]) - cm))/(c[mask].shape[0]-1)
      for k in range(3):
        if S[k,k] == 0: S[k,k]=1.e-4
      Sinv = np.linalg.inv(S)
      #correct centroid's position
      val = []
      for idx in c[mask]:
        idx = np.array(idx) 
        d = (idx - cm).dot(Sinv).dot((idx-cm).transpose())
        d =  math.sqrt(d)
        val.append(d)
      index = val.index(min(val))
      new_cm = np.array(c[mask][index])
      del val
      #Sigmainv.append(np.triu(Sinv))
      invCovM.append(Sinv)
      centroids.append(cm)
      c_centroids.append(new_cm)
      # Calculate the offset for each voxel
      off =[]
      for idx in c[mask]:
        idx = np.array(idx)
        d = (idx - new_cm).dot(Sinv).dot((idx-new_cm).transpose())
        d =  math.sqrt(d)
        off.append(d)
      offsets[mask] = off
  #Get the probability of every active voxel
      p = gauss_pdf(c[mask],fixCovMinv,new_cm)
      prob[mask] = p

  return centroids, offsets, invCovM, c_centroids, prob

