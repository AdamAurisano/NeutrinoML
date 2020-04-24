'''
Module for retrieving PyTorch network architectures
'''

from .minkowski_seg import MinkowskiSeg
from .minkowski_class import MinkowskiClass, Minkowski2StackClass

_models = { 'MinkowskiSeg':   MinkowskiSeg,
            'MinkowskiClass': MinkowskiClass,
            'Minkowski2StackClass': Minkowski2StackClass }

def get_model(name, **model_args):
    
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception(f'Model {name} unknown.')
