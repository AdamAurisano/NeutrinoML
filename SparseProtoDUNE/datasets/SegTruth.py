'''Code to produce per-hit segmentation ground truth'''
import logging, numpy as np
from particle import PDGID, Particle

#def SegTruth(pdg, trackId, energy, proc, end_proc, co):
def SegTruth(pdg, trackId, energy, proc, end_proc):
    y_true = np.zeros([len(pdg),10], dtype=np.float32)
    noise_mask = [ (len(p) > 0) for p in pdg ]
   
   #nuclei_list = []
   # nproc_list = []
   # n_end_proc_list = []
   # c = []

   # p_proc = []
   # p_end_proc = [] 

   # nuc_proc = []
   # nuc_end_proc = []
   # nuc_pdg = []
   # nuc_c = []

   # sh_proc = []
   # sh_end_proc = []
   # sh_c = []
   # sh_tr = []

    IDs = np.zeros(len(pdg))
    for idx in range(len(pdg)):
    ## getting per space point Id
        e = np.array(energy[idx])
        _id = np.array(trackId[idx])
        if (_id.shape[0] ==0): continue
        #if (_id.shape[0]>0 and _id.shape[0]<= 2):
           # sp_id=_id[np.where(e==e.max())].item()
        elif _id.shape[0] > 0:
            ids, ct = np.unique(_id,return_counts=True)
            e_sum = []
            for k in ids:
                e_sum.append(e[np.where(_id==k)].sum())
            sp_id=ids[e_sum.index(max(e_sum))]

    ## Getting per space point ground truth
    #['shower','delta','diffuse','kaons','michel','hip','mu','pi','unknown'] 
         
        for jdx in range(len(pdg[idx])): #loop over space points
            # Showers from pi0
            if trackId[idx][jdx] <0:
                y_true[idx,0] += energy[idx][jdx]
            # Diffuse
            elif pdg[idx][jdx] == 97: 
                y_true[idx,2] += energy[idx][jdx]
            elif  (pdg[idx][jdx] != 2212 and PDGID(pdg[idx][jdx]).is_nucleus == True and proc[idx][jdx] == 'neutronInelastic' and (end_proc[idx][jdx] == 'ionIoni' or end_proc[idx][jdx] == 'hIoni')):
                y_true[idx,2] += energy[idx][jdx]
            # Delta  
            elif pdg[idx][jdx] == 95 or  pdg[idx][jdx] == 96 or pdg[idx][jdx] == 98:
                y_true[idx,1] += energy[idx][jdx]
            # Michel
            elif abs(pdg[idx][jdx]) == 99:
                y_true[idx,4] += energy[idx][jdx]
            # Kaon
            elif abs(pdg[idx][jdx]) == 321:
                 y_true[idx,3] += energy[idx][jdx]
            # Muon
            elif abs(pdg[idx][jdx]) == 13:
                y_true[idx,6] += energy[idx][jdx]
            # Pion
            elif abs(pdg[idx][jdx]) == 211:
                y_true[idx,7] += energy[idx][jdx]
            #  hip ->  Just Protons
            elif pdg[idx][jdx] == 2212:
                y_true[idx,5] += energy[idx][jdx]
                #p_proc.append(proc[idx][jdx])
                #p_end_proc.append(end_proc[idx][jdx])
            # hip 
            #elif  (pdg[idx][jdx] != 2212 and PDGID(pdg[idx][jdx]).is_nucleus == True and end_proc[idx][jdx] == 'hIoni'): 
                #y_true[idx,5] += energy[idx][jdx]
                #nuclei_list.append(pdg[idx][jdx])
                #nproc_list.append(proc[idx][jdx])
                #n_end_proc_list.append(end_proc[idx][jdx])
                #c.append(co[idx])
            # Nuclei -> unknown
            #elif  (pdg[idx][jdx] != 2212 and PDGID(pdg[idx][jdx]).is_nucleus == True and end_proc[idx][jdx] == 'hIoni'): 
            elif  (pdg[idx][jdx] != 2212 and PDGID(pdg[idx][jdx]).is_nucleus == True):
                y_true[idx,8] += energy[idx][jdx]
                #nuc_pdg.append(pdg[idx][jdx])
                #nuc_proc.append(proc[idx][jdx])
                #nuc_end_proc.append(end_proc[idx][jdx])
                #nuc_c.append(co[idx])
            # rest of electrons 
            elif abs(pdg[idx][jdx]) == 11 and (proc[idx][jdx] == 'compt' or proc[idx][jdx] == 'phot'):
                y_true[idx,2] += energy[idx][jdx]
            elif abs(pdg[idx][jdx]) == 11: 
                y_true[idx,0] += energy[idx][jdx]
               # sh_proc.append(proc[idx][jdx])
               # sh_end_proc.append(end_proc[idx][jdx])
               # sh_c.append(co[idx])
               # sh_tr.append(trackId[idx][jdx])

            elif abs(pdg[idx][jdx]) ==3222: continue
            elif abs(pdg[idx][jdx]) ==3112: continue
            else: raise Exception('Unrecognised particle found in ground truth!', f'Truth information not recognised! Particle is {Particle.from_pdgid(pdg[idx][jdx]).name} ({pdg[idx][jdx]})')
        IDs[idx] = sp_id

    return noise_mask, y_true, IDs #, sh_proc, sh_end_proc, sh_c, sh_tr   #, nuclei_list, nproc_list, n_end_proc_list, c, nuc_proc, nuc_end_proc, nuc_pdg, nuc_c 
