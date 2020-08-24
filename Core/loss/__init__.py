'''
Module to load both standard and custom loss functions
'''

def get_loss(loss, **params):
  if loss == 'categorical_cross_entropy':
    from .loss import categorical_cross_entropy
    return categorical_cross_entropy
  else:
    from torch import nn
    return getattr(nn.modules.loss, loss)(**params)

