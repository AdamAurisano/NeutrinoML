'''Code to produce per-hit segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle

def GetShowers(pdg, trackId):
  em_particles = [ abs(it1_id) for it2_pdg, it2_id in zip(pdg, trackId) for it1_pdg, it1_id in zip(it2_pdg, it2_id) if abs(it1_pdg) == 11 or it1_pdg == 22 ]
  unique = set(em_particles)
  count = { u: em_particles.count(u) for u in unique }
  return tuple(key for key, val in count.items() if val > 10 )

def SegTruth(pdg, trackId, process, energy):

  y_true = np.zeros([len(pdg),7], dtype=np.float32)
  noise_mask = [ (len(p) > 0) for p in pdg ]

  # Figure out which EM hits should be considered showers
  showers = GetShowers(pdg, trackId)

  for idx in range(len(pdg)):
    for jdx in range(len(pdg[idx])):
      # Showers
      if (abs(pdg[idx][jdx]) == 11 or pdg[idx][jdx] == 22):
        if abs(trackId[idx][jdx]) in showers:
          y_true[idx,0] += energy[idx][jdx] # Shower if track ID produced more than ten hits
        else:
          y_true[idx,1] += energy[idx][jdx] # Diffuse energy deposition if ten or fewer hits
      # Diffuse energy depositions
      elif trackId[idx][jdx] < 0 or process[idx][jdx] == b'neutronInelastic' or process[idx][jdx] == b'nCapture':
        y_true[idx,1] += energy[idx][jdx]
      elif trackId[idx][jdx] > 0:
        # Kaons 
        if abs(pdg[idx][jdx]) == 321:
          y_true[idx,2] += energy[idx][jdx]
        # Michel electrons + positrons
        elif pdg[idx][jdx] == 98 or pdg[idx][jdx] == 99:
          y_true[idx,3] += energy[idx][jdx]
        # Highly ionising particles
        elif PDGID(pdg[idx][jdx]).is_nucleus == True or 'Sigma' in Particle.from_pdgid(pdg[idx][jdx]).name:
          y_true[idx,4] += energy[idx][jdx]
        # Muons
        elif abs(pdg[idx][jdx]) == 13: # or abs(pdg[idx][jdx]) == 211: 
          y_true[idx,5] += energy[idx][jdx]
        # Pions
        elif abs(pdg[idx][jdx]) == 211:
          y_true[idx,6] += energy[idx][jdx]
        else: raise Exception('Unrecognised particle found in ground truth!')#f'Truth information not recognised! Particle is {Particle.from_pdgid(__pdg).name} ({__pdg})')
      else: raise Exception('Unrecognised particle found in ground truth!')#f'Truth information not recognised! PDG is {Particle.from_pdgid(__pdg).name} ({__pdg})')

  return noise_mask, y_true

