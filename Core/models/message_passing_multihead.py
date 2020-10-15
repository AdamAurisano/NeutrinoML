from .message_passing import GNNSegmentClassifier
import torch
import torch_geometric as tg
from torch import nn

class MultiHead(nn.Module):
    def __init__(self, classes, input_dim, hidden_dim=8, n_iters=3, A=nn.Tanh, **kwargs):
        super(MultiHead, self).__init__()
        self.net = nn.ModuleList([ GNNSegmentClassifier(input_dim, hidden_dim, n_iters, A) for i in range(classes) ])

    def forward(self, x):
        return torch.stack([net(x) for net in self.net]).transpose(0,1)