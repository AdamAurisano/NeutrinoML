import torch_geometric.transforms as T

class FeatureNorm(T.BaseTransform):

  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, data):
    for p in [ "u", "v", "y" ]:
      key = f"x_{p}"
      data[key] = (data[key]-self.mean[None,:]) / self.std[None,:]
    return data

