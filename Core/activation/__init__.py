'''
Module to load activation functions
'''
from .minkowski_activation import MinkowskiActivation

def get_activation(activation, **params):

  if True:
    from torch import nn
    a = getattr(nn, activation)
    return a(**params[activation]) if activation in params else a()

