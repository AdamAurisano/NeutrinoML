'''
Module to load both standard and custom loss functions
'''

def get_loss(func, **params):
  if func == 'categorical_cross_entropy':
    from .loss import categorical_cross_entropy
    return categorical_cross_entropy
  if func == 'cross_entropy':
    from .loss import cross_entropy
    return cross_entropy
  if func == 'generalized_dice':
    from .loss import generalized_dice
    return generalized_dice
  if func == "twoloss":
    from .twoloss import double_loss
    return double_loss(**params)
  if func == "categorical_focal_loss":
    from .loss import categorical_focal_loss
    return categorical_focal_loss
  if func == 'focal_loss':
    from .loss import focal_loss
    return focal_loss
  else:
    from torch import nn
    return getattr(nn.modules.loss, func)(**params)

