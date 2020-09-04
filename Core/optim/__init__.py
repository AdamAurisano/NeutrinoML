'''
Module to load optimizers
'''

def get_optim(model_params, optimizer, **params):

  if optimizer == 'Fromage':
    from .fromage import Fromage
    ret = Fromage
  elif optimizer == 'Ranger':
    from .ranger import Ranger
    ret = Ranger
  else:
    import torch.optim
    ret = getattr(torch.optim, optimizer)

  return ret(model_params, **params[optimizer])

