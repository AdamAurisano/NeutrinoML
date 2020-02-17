import torch
import uuid
import uproot
import numpy as np
from PIL import Image
from particle import PDGID
from particle import Particle
from collections import Counter
import matplotlib.pyplot as plt

class Pixelmap():
    def __init__(self,coo=None, pdg=None, trackId=None, energy=None, process=None, parents=None, pixelval=None):
        self.coo      = np.array(coo)
        self.pdg      = np.array(pdg)
        self.trackId  = np.array(trackId)
        self.energy   = np.array(energy)
        self.process  = np.array(process)
        self.parents  = np.array(parents)
        self.pixelval = np.array(pixelval)

## divide the dataset into to subsets corresponding to each side of the detector ( right and left)
    def get_volumes(self):
        tpc = self.coo[:,2]
        #print(tpc)
        cut_vol1 = (tpc==1) | (tpc==5) | (tpc==9)
        cut_vol2 = (tpc==2) | (tpc==6) | (tpc==10)

        ret = [ { 'Coordinates':self.coo[c,:2], 'PDG':self.pdg[c], 'TrackId':self.trackId[c],
                  'Process':self.process[c], 'Energy':self.energy[c], 'PixelValue':self.pixelval[c] }
                for c in [ cut_vol1, cut_vol2 ] ]

        return ret
