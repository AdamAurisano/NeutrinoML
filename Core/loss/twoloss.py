import torch, torch.nn as nn, torch.nn.functional as F

class double_loss(nn.Module):
  def __init__(self, b_weight, c_weight):
    super(double_loss, self).__init__()
    self.b_weight = b_weight
    self.c_weight = c_weight

  def forward(self, pred, true):
    B, C = pred
    mask = (true != 0)
    binary_true = (true > 0)
    print(binary_true)
    print(B)
    print(C)
    binary_loss = F.binary_cross_entropy(B, binary_true.float(), self.b_weight)
    class_loss = F.nll_loss(torch.log(C[mask]), true[mask]-1, self.c_weight)
    print(binary_loss, class_loss)
    return binary_loss + class_loss
