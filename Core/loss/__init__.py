'''
Module to load both standard and custom loss functions
'''

def get_loss(loss, **params):
  if loss == 'categorical_cross_entropy':
    from .loss import categorical_cross_entropy
    return categorical_cross_entropy
  if loss == 'generalized_dice':
    from .loss import generalized_dice
    return generalized_dice
  else:
    from torch import nn
    return getattr(nn.modules.loss, loss)(**params)

