'''
Module to load both standard and custom loss functions
'''

def get_loss(func, **params):
  if func == 'categorical_cross_entropy':
    from .loss import categorical_cross_entropy
    return categorical_cross_entropy
  if func == 'generalized_dice':
    from .loss import generalized_dice
    return generalized_dice
  else:
    from torch import nn
    return getattr(nn.modules.loss, func)(**params)

