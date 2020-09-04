'''
Module to load activation functions
'''
from .minkowski_activation import MinkowskiActivation

def get_activation(activation, **params):

  if activation == 'mish':
    from .mish import Mish
    return Mish()
  else:
    from torch import nn
    a = getattr(nn, activation)
    return a(**params[activation]) if activation in params else a()

