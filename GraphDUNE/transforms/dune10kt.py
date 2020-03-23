
'''
Custom transforms for DUNE 10kt detector
'''
import torch, numpy as np

class NormPosDUNE(object):
  '''Normalise position within DUNE detector geometry'''
  def __init__(self):
    self.add = torch.tensor([750,600,0])
    self.mult = torch.tensor(1./6000.) # ranges are 1500, 1200, 6000 but want to maintain isotropy
  def __call__(self, data):
    data.pos += self.add[None,:].to(data.pos.device)
    data.pos *= self.mult[None,None].to(data.pos.device)

class NormFeatDUNE(object):
  '''Normalise features for DUNE spacepoint graph'''
  def __init__(self):
    self.mult = torch.tensor([1./1.e6,1./400.,1./4500.,1./1.e6,1./400.,1./4500.,1./1.e6,1./480.,1./4500.,1./1.e6], dtype=torch.float)
  def __call__(self, data):
    data.x *= self.mult[None,:].to(data.x.device)

class NormDUNE(object):
  '''Normalise features and positions for DUNE spacepoint graph'''
  def __init__(self):
    self.xtrans = NormFeatDUNE()
    self.postrans = NormPosDUNE()
  def __call__(self, data):
    self.xtrans(data)
    self.postrans(data)

