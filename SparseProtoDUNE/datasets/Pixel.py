from .Pixelmap import *
from particle import Particle

class Pixel():
    def GetEnergy(En, trackId, trueEmshowers, GT):
        for idx in range(En.shape[0]):
            for jdx in range(len(En[idx])):
                if abs(trackId[idx][jdx]) in trueEmshowers:
                    GT[idx,2] += En[idx][jdx] / sum(En[idx])
        return GT
        
    def GetEnergy1(En, trackId, Emactivity, GT):
        for idx in range(En.shape[0]):
            for jdx in range(len(En[idx])):
                if abs(trackId[idx][jdx]) in Emactivity:
                    GT[idx,1] += En[idx][jdx] / sum(En[idx])
        return GT

    def classifier(dic):
      
        coo      = dic['Coordinates']
        pdg      = dic['PDG']
        trackId  = dic['TrackId']
        process  = dic['Process']
        energy   = dic['Energy']
        pixelval = dic['PixelValue']

        EmtrackId = []
        GT = np.zeros([coo.shape[0],7]).astype(np.float32)
        for idx in range(coo.shape[0]):
        #if len(pdg[idx]) == 0:
        #    continue
            if len(pdg[idx]) != 0:
#                norm = 1/sum(energy[idx])
                for jdx in range(len(pdg[idx])):
                    #kaons 
                    if  abs(pdg[idx][jdx]) == 321:
                         GT[idx,0] += energy[idx][jdx]
                    # Neutrons
                    if process[idx][jdx] == b'neutronInelastic' or process[idx][jdx] == b'nCapture':
                        GT[idx,1] += energy[idx][jdx]
                    # Showers
                    elif (abs(pdg[idx][jdx]) == 11 or pdg[idx][jdx] == 22):
                        EmtrackId.append(abs(trackId[idx][jdx])) ## not all are EM Showers
                    # EM activity
                    elif trackId[idx][jdx] < 0:
                        GT[idx,1] += energy[idx][jdx]
                    elif trackId[idx][jdx] > 0:
                    # Michel electrons + positrons
                        if pdg[idx][jdx] == 98 or pdg[idx][jdx] == 99:
                            GT[idx,6] += energy[idx][jdx]
                    # Highly ionising particles
                        elif PDGID(pdg[idx][jdx]).is_nucleus == True or 'Sigma' in Particle.from_pdgid(pdg[idx][jdx]).name:
                            GT[idx,3] += energy[idx][jdx]
                    # Muons
                        elif abs(pdg[idx][jdx]) == 13: # or abs(pdg[idx][jdx]) == 211: 
                            GT[idx,4] += energy[idx][jdx]
                    # Pions
                        elif abs(pdg[idx][jdx]) == 211:
                            GT[idx,5] += energy[idx][jdx]
                 #else: print(f'Truth information not recognised! Particle is {Particle.from_pdgid(__pdg).name} ({__pdg})')
                #else: print(f'Truth information not recognised! PDG is {Particle.from_pdgid(__pdg).name} ({__pdg})')
#            GT[idx,:] *= norm    
        ## Aditional steps to disambiguate Em ativity and Em showers 

        trIds = Counter(EmtrackId)   ## dictionary of tracks and number of hits for all e+ e- and photons 
        trueEmshowers = []
        Emactivity = []
        for Id in trIds:
            if trIds.get(Id) > 10:
                trueEmshowers.append(Id)
            else:
                 Emactivity.append(Id)

        NewGT = Pixel.GetEnergy(energy, trackId, trueEmshowers, GT)
        NewGT = Pixel.GetEnergy1(energy, trackId, Emactivity, NewGT)

        #dic = {'MIP': MIP, 'HIP': HIP, 'EmtracksId': EmtrackId, 'Me':Me, 'neutrons': neutrons, 'Noise': Noise,
        #       'EmActivity':EmAct, 'All_Em': trueEmshowers} 

        coo = coo.astype(np.int32)
        pixelval *= 0.0001

        dic = {'GroundTruth': NewGT, 'Coordinates': coo, 'PixelValue': pixelval}
        return (dic)
