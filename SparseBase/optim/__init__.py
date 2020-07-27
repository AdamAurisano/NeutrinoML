'''
Module to load optimizers
'''

def get_optim(model_params, optimizer, **params):

  # Not hooking in Ranger just yet, but we should
  if True:
    import torch.optim
    return gettattr(torch.optim, optimizer)(
      model_params, **params[optimizer])

