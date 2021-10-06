'''Code to produce per-hit segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle

def SegTruth(pdg, trackId, energy):
    y_true = np.zeros([len(pdg),9], dtype=np.float32)
    noise_mask = [ (len(p) > 0) for p in pdg ]

    IDs = np.zeros(len(pdg))
    for idx in range(len(pdg)):
    ## getting per space point Id
        e = np.array(energy[idx])
        _id = np.array(trackId[idx])
        if (_id.shape[0] ==0): continue
        if (_id.shape[0]>0 and _id.shape[0]<= 2):
            sp_id=_id[np.where(e==e.max())].item()
        elif _id.shape[0] > 2:
            ids, ct = np.unique(_id,return_counts=True)
            e_sum = []
            for k in ids:
                e_sum.append(e[np.where(_id==k)].sum())
            sp_id=ids[e_sum.index(max(e_sum))]

    ## Getting per space point ground truth
     #['shower','delta','diffuse','kaons','michel','hip','mu','pi','electros']
    
        for jdx in range(len(pdg[idx])): #loop over each pixel
        # showers from pi0
            if trackId[idx][jdx] <0:
                y_true[idx,0] += energy[idx][jdx]
            #Electrons
            elif abs(pdg[idx][jdx]) == 11 or abs(pdg[idx][jdx]) == 22:
                y_true[idx,8] += energy[idx][jdx]
            #diffuse
            elif abs(pdg[idx][jdx]) == 97:
                y_true[idx,2] += energy[idx][jdx]
            #Delta  
            elif pdg[idx][jdx] == 98:
                y_true[idx,1] += energy[idx][jdx]
             #michel
            elif abs(pdg[idx][jdx]) == 99:
                y_true[idx,4] += energy[idx][jdx]
            #Kaon
            elif abs(pdg[idx][jdx]) == 321:
                 y_true[idx,3] += energy[idx][jdx]
            #muon
            elif abs(pdg[idx][jdx]) == 13:
                y_true[idx,6] += energy[idx][jdx]
            #pion
            elif abs(pdg[idx][jdx]) == 211:
                y_true[idx,7] += energy[idx][jdx]
            #protons and nuclei
            elif PDGID(pdg[idx][jdx]).is_nucleus == True:
                y_true[idx,5] += energy[idx][jdx]
            else: raise Exception('Unrecognised particle found in ground truth!', f'Truth information not recognised! Particle is {Particle.from_pdgid(pdg[idx][jdx]).name} ({pdg[idx][jdx]})')
        IDs[idx] = sp_id
    return noise_mask, y_true, IDs 
