'''Load network architecture'''

from .simple import TauRNN

_models = { 'TauRNN': TauRNN }

def get_model(name, **model_args):
    
    if name in _models:
        return _models[name](**model_args)
    else:
        raise Exception(f'Model {name} unknown.')

