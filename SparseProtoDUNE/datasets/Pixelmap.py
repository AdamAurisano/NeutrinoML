import os
import torch
import uuid
import uproot
import numpy as np
from PIL import Image
from particle import PDGID
from particle import Particle 
from collections import Counter
import matplotlib.pyplot as plt
from sympy import pretty_print as pp, latex
from sympy.abc import a, b, n

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
    #_scoor1, _scoor2         = [], []
    #_spdg1, _spdg2           = [], []
    #_strackId1, _strackId2   = [], []
    #_spdg1, _spdg2           = [], []
    #_sparents1, _sparents2   = [], []
    #_sprocess1, _sprocess2   = [], []
    #_spixelval1, _spixelval2 = [], []
    #_senergy1, _senergy2      = [], []

    tpc = self.coo[:,2]

    #print(tpc)

    cut_vol1 = (tpc==1) | (tpc==5) | (tpc==9)
    cut_vol2 = (tpc==2) | (tpc==6) | (tpc==10)

    ret = [ { 'Coordinates':self.coo[c,:2], 'PDG':self.pdg[c], 'TrackId':self.trackId[c],
              'Process':self.process[c], 'Energy':self.energy[c], 'PixelValue':self.pixelval[c] }
            for c in [ cut_vol1, cut_vol2 ] ]

    return ret

    #Vol1 = { 'Coordinates':self.coo[cut_vol1,:2], 'PDG':self.pdg[cut_vol1], 'TrackId':self.trackId[cut_vol1],
    #         '}

    #for _coo, _parents, _pdg, _trackId, _process, _energy, _pixelval in zip(self.coo, self.parents, self.pdg, self.trackId,
    #                                                    self.process, self.energy, self.pixelval):
    #    if _coo[2] == 1 or _coo[2] == 5 or _coo[2]==9:
    #        _scoor1.append(_coo[:2])
    #        _strackId1.append(_trackId)
    #        _spdg1.append(_pdg)
    #        _sprocess1.append(_process)
    #        _sparents1.append(_parents)
    #        _spixelval1.append(_pixelval)
    #        _senergy1.append(_energy)

    #    elif _coo[2] == 2 or _coo[2] == 6 or _coo[2]==10:
    #        _scoor2.append(_coo[:2])
    #        _strackId2.append(_trackId)
    #        _spdg2.append(_pdg)
    #        _sprocess2.append(_process)
    #        _sparents2.append(_parents)
    #        _spixelval2.append(_pixelval)
    #        _senergy2.append(_energy)

                
    #Vol1 = {'Coordinates':_scoor1, 'PDG': _spdg1, 'TrackId': _strackId1, 'Parents': _sparents1, 'Process': _sprocess1,
    #            'Energy':_senergy1, 'PixelValue': _spixelval1}              

    #Vol2 = {'Coordinates':_scoor2, 'PDG': _spdg2, 'TrackId': _strackId2, 'Parents': _sparents2, 'Process': _sprocess2,
    #            'Energy':_senergy2, 'PixelValue': _spixelval2}
    #return (Vol1, Vol2)
