'''
Module for retrieving PyTorch network architectures
'''

from .seg import *
from .minkowski_seg import MinkowskiSeg
from .minkowski_class import MinkowskiClass, Minkowski2StackClass
from.nova_mobilenet import MobileNet

_models = { 'MinkowskiSeg':   MinkowskiSeg,
            'MinkowskiClass': MinkowskiClass,
            'Minkowski2StackClass': Minkowski2StackClass,
            'MobileNet': MobileNet,
            'SparseSegmentation': SparseSegmentation}

def get_model(name, **model_args):
    
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception(f'Model {name} unknown.')



