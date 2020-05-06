'''
Module for holding onto PyTorch Geometric network architectures
'''

from .pointnet import *
from .message_passing import *

_models = { 'pointnet': PointNet,
            'message': GNNSegmentClassifier }

def get_model(name, **model_args):
  if name in _models:
    return _models[name](**model_args)
  else:
    raise Exception(f'Model {name} unknown.')

