'''
Module to load both standard and custom loss functions
'''

def get_loss(loss):
  if loss == 'categorical_cross_entropy':
    from .loss import categorical_cross_entropy
    return categorical_cross_entropy
  elif loss == 'cross_entropy_loss':
    from torch.nn import CrossEntropyLoss
    return CrossEntropyLoss
  else:
    from torch import nn
    return getattr(nn.modules.loss, loss)

