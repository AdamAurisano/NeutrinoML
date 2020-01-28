'''
Module for holding onto PyTorch network architectures
'''

from .seg import *

_models = { 'SparseSegmentation': SparseSegmentation }

def get_model(name, **model_args):

  if name in _models:
    return _models[name](**model_args)
  else:
    raise Exception(f'Model {name} unknown.')

