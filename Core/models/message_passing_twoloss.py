"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch, torch.cuda as tc
import torch.nn as nn
from torch_scatter import scatter_add

from .message_passing_multihead_deep import CustomLinear, MultiClassNodeNetwork

class MultiClassEdgeNetwork(nn.Module):
  def __init__(
    self,
    input_dim=2,
    output_dim=8,
    nclasses=4,
    hidden_activation=nn.Tanh,
    how='standard',
    stacks=True,
    edgefeats=False,
    **kwargs,
  ):

    super(MultiClassEdgeNetwork, self).__init__()
    self.nclasses=nclasses
    self.edgefeats=edgefeats
    if not stacks: how = 'standard'
    output = 1 if stacks else nclasses
    self.radlen = nn.Parameter(torch.ones(nclasses))
    in_size = input_dim * 2
    if edgefeats: in_size += 2
    self.net1 = nn.Sequential(
      CustomLinear(in_size, output_dim, nclasses, how),
      hidden_activation()
    )
    self.binary_net = nn.Sequential(
      nn.Linear(output_dim * nclasses, 1),
      nn.Sigmoid(),
    )  
    self.class_net = nn.Sequential(
      CustomLinear(output_dim, output, nclasses, how),
      hidden_activation(),
      nn.Softmax(dim=1),
    )

  def calc_eloss(self, data, distance):
    integral = data.x[data.edge_index, 7]
    loss = integral.max(dim=0).values / integral.min(dim=0).values
    decay = torch.exp(-distance.unsqueeze(-1).repeat(1,self.nclasses)/self.radlen)
    return loss.unsqueeze(-1).repeat(1, self.nclasses) * decay

  def calc_edge_feats(self, data):
    row,col = data.edge_index
    wire = 5 * (data.x[row,1] - data.x[col,1])
    time = 0.8 * (data.x[row,2] - data.x[col,2])
    distance = torch.sqrt(torch.square(wire) + torch.square(time))
    dqdx = (data.x[row,7]+data.x[col,7])/(2*distance)
    dqdx = dqdx.unsqueeze(-1).repeat(1, self.nclasses)
    eloss = self.calc_eloss(data, distance)
    return torch.stack([dqdx,eloss], dim=-1)

  def forward(self, data):
    row,col = data.edge_index
    edge_feats = [ data.m[col], data.m[row] ]
    if (self.edgefeats):
      edge_feats.append(self.calc_edge_feats(data))
    A = torch.cat(edge_feats, dim=-1).detach()
    A = self.net1(A)
    B = self.binary_net(A.contiguous().view(A.shape[0], -1)).unsqueeze(dim=1)
    C = self.class_net(A)
    C = B * C
    return B, C

class GNNTwoLoss(nn.Module):
  def __init__(self, input_dim=2, output_dim=4, hidden_dim=8, n_iters=3,
               hidden_activation=nn.Tanh, how='standard',
               stacks=True, edgefeats=False, **kwargs):
    super(GNNTwoLoss, self).__init__()
    self.n_iters = n_iters
    self.nclasses = output_dim
    self.stacks = stacks
    if not stacks: how = 'standard'
    self.input_network = nn.Sequential(
      CustomLinear(input_dim, hidden_dim, output_dim, how),
      hidden_activation(),
    )
    # Setup the edge network
    self.edge_network = MultiClassEdgeNetwork(
      input_dim + hidden_dim,
      hidden_dim,
      output_dim,
      hidden_activation,
      how,
      stacks,
      edgefeats,
    )
    # Setup the node layers
    self.node_network = MultiClassNodeNetwork(
      input_dim + hidden_dim,
      hidden_dim,
      output_dim,
      hidden_activation,
      how,
    )

  def forward(self, data):
    """Apply forward pass of the model"""
    global dev
    dev = data.x.device
    # mem("at model start")
    X = data.x.unsqueeze(1).repeat(1, self.nclasses, 1) if self.stacks else data.x.clone()
    # Apply input network to get hidden representation
    H = self.input_network(X)
    # Shortcut connect the inputs onto the hidden representation
    data.m = torch.cat([H, X], dim=-1)
    # Loop over iterations of edge and node networks
    # mem("before iterating")

    for i in range(self.n_iters):
      # Apply edge network, update edge_attrs
      _, data.edge_attr = self.edge_network(data)
      # mem("after edge network")
      # Apply node network
      H = self.node_network(data)
      # mem("after node network")
      # Shortcut connect the inputs onto the hidden representation
      data.m = torch.cat([H, X], dim=-1)
      # mem(f"after iteration {i}")

    # Apply final edge network
    B, C = self.edge_network(data)
    return B.squeeze(), C.squeeze()

