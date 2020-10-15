import torch

def categorical_cross_entropy(y_pred, y_true):
  y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
  # Normalise the loss!
  loss = -(y_true * torch.log(y_pred))
  weights = torch.zeros(y_true.shape[1]).to(y_true.device)
  class_sum = y_true.sum(dim=0)
  mask = (class_sum>0)
  weights[mask] = y_true[:,mask].shape[0]/(y_true.shape[1]*class_sum[mask])
  weighted_loss = weights[None,:] * loss
  return weighted_loss.sum(dim=1).mean()


def generalized_dice(y_pred, y_true):
  weights = torch.zeros(y_true.shape[1]).to(y_true.device)
  class_sum = y_true.sum(dim=0)
  mask = (class_sum>0)
  weights[mask] = 1/y_true[:,mask].sum(dim=0)**2
  num = y_pred*y_true
  num = (weights*num)[:,mask].sum(dim=0)
  den = (y_pred + y_true)
  den = (weights*den)[:,mask].sum(dim=0)
  
  
  dice_coef = 2*num.sum()/den.sum()
  val = 1 -dice_coef
  return val
