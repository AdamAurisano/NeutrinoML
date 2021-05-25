'''Code to produce per-hit segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle

def GetShowers(pdg, trackId):
  em_particles = [ abs(it1_id) for it2_pdg, it2_id in zip(pdg, trackId) for it1_pdg, it1_id in zip(it2_pdg, it2_id) if abs(it1_pdg) == 11 or it1_pdg == 22 ]
  unique = set(em_particles)
  count = { u: em_particles.count(u) for u in unique }
  return tuple(key for key, val in count.items() if val > 45)

def SegTruth(pdg, trackId, process, energy):

  y_true = np.zeros([len(pdg),8], dtype=np.float32)
  proc   = np.empty([len(pdg),8], dtype=object)
  noise_mask = [ (len(p) > 0) for p in pdg ]
 
  # Figure out which EM hits should be considered showers
  showers = GetShowers(pdg, trackId)

  #['shower','delta','diffuse','kaons','michel','hip','mu','pi'] 
  for idx in range(len(pdg)):
    if len(pdg[idx]) > 0:
      for jdx in range(len(pdg[idx])): #loop over each pixel 
        # Delta rays
        if pdg[idx][jdx] == 97:
          y_true[idx,1] += energy[idx][jdx]
          proc[idx,1] =  str(process[idx][jdx])
        # Showers
        elif abs(pdg[idx][jdx]) == 11:  
          if abs(trackId[idx][jdx]) in showers:
            if process[idx][jdx]=='phot' or process[idx][jdx]=='eIoni' or process[idx][jdx]=='conv' or process[idx][jdx]=='compt':
              y_true[idx,0] += energy[idx][jdx]
              proc[idx,0] = str(process[idx][jdx])
        # Delta rays
            elif process[idx][jdx]=='muIoni':
              y_true[idx,1] += energy[idx][jdx]
              proc[idx,1] =  str(process[idx][jdx])
        # Diffuse energy depositions
            else: 
              y_true[idx,2] += energy[idx][jdx] # Diffuse energy deposition
              proc[idx,2] = str(process[idx][jdx])
        # kaons 
        elif abs(pdg[idx][jdx]) == 321:
          y_true[idx,3] += energy[idx][jdx]
          proc[idx,3] =  str(process[idx][jdx])
        # Michel electrons + positrons
        elif pdg[idx][jdx] == 98 or pdg[idx][jdx] == 99:
          y_true[idx,4] += energy[idx][jdx]
          proc[idx,4] = str(process[idx][jdx])
        # Highly ionising particles
        elif PDGID(pdg[idx][jdx]).is_nucleus == True or 'Sigma' in Particle.from_pdgid(pdg[idx][jdx]).name:
          y_true[idx,5] += energy[idx][jdx]
          proc[idx,5] = str(process[idx][jdx])
        # Muons
        elif abs(pdg[idx][jdx]) == 13:  
          y_true[idx,6] += energy[idx][jdx]
          proc[idx,6] = str(process[idx][jdx])
        # Pions
        elif abs(pdg[idx][jdx]) == 211:
          y_true[idx,7] += energy[idx][jdx]
          proc[idx,7] = str(process[idx][jdx])
       # else: raise Exception('unrecognize particle', pdg[idx][jdx])
        else: raise Exception('Unrecognised particle found in ground truth!', f'Truth information not recognised! Particle is {Particle.from_pdgid(pdg[idx][jdx]).name} ({pdg[idx][jdx]})')
      #else: raise Exception('Unrecognised particle found in ground truth!', f'Truth information not recognised! PDG is {Particle.from_pdgid(pdg[idx][jdx]).name} ({pdg[idx][jdx]})')

  return noise_mask, y_true, proc 

