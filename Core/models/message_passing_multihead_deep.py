"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch, torch.cuda as tc
import torch.nn as nn
from torch_scatter import scatter_add

def mem(name):
  global dev
  print(
    "Memory consumption",
    name,
    "is",
    float(tc.memory_allocated(dev)) / float(1073741824),
    "GB",
  )

def tsize(t):
  return t.element_size() * t.nelement()

class FlattenedLinear(nn.Module):
  '''Module that flattens the last two dimensions of a tensor into a single
  dimension before performing a linear convolution, and then unflattens
  afterwards'''
  def __init__(self, in_feats, out_feats, nclasses, **kwargs):
    super(FlattenedLinear, self).__init__()
    self.shape_flattened = [nclasses * in_feats]
    self.shape_out = [nclasses,out_feats]
    self.net = nn.Linear(nclasses * in_feats, nclasses * out_feats)

  def forward(self, x):
    shape = list(x.shape[:-2])
    x = torch.reshape(x, shape + self.shape_flattened)
    x = self.net(x)
    return torch.reshape(x, shape + self.shape_out)

class ClasswiseLinear(nn.Module):
  '''Module that performs a separate set of linear convolutions along the
  second-to-last dimension of a tensor'''
  def __init__(self, in_feats, out_feats, nclasses, **kwargs):
    super(ClasswiseLinear, self).__init__()
    self.out_feats = out_feats
    self.nclasses = nclasses
    self.net = nn.ModuleList([nn.Linear(in_feats, out_feats) for i in range(nclasses)])

  def forward(self, x):
    if x.ndim != 3:
      raise Exception(f'This module only works for 3D tensors, but this tensor has {x.ndim} dimensions.')
    if x.shape[1] != len(self.net):
      raise Exception(f'Tensor size in penultimate dimension ({x.shape[1]}) does not match number of classes ({len(self.net)}).')
    return torch.cat(
      [ net(i) for net, i in zip(self.net, torch.tensor_split(x, [*range(1,self.nclasses)], dim=1))],
      dim=1,
    )

class CustomLinear(nn.Module):
  '''Switch out different ways of doing linear convolutions!'''
  def __init__(self, in_feats, out_feats, nclasses, how, **kwargs):
    super(CustomLinear, self).__init__()
    if how == 'standard':
      self.net = nn.Linear(in_feats, out_feats)
    elif how == 'flatten':
      self.net = FlattenedLinear(in_feats, out_feats, nclasses)
    elif how == 'classwise':
      self.net = ClasswiseLinear(in_feats, out_feats, nclasses)
    else: raise Exception(f'Linear method "{how}" not recognised!')

  def forward(self, x):
    return self.net(x)

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
    if edgefeats: self.radlen = nn.Parameter(torch.ones(nclasses))
    in_size = input_dim * 2
    if edgefeats: in_size += 2
    self.net = nn.Sequential(
      CustomLinear(in_size, output_dim, nclasses, how),
      hidden_activation(),
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
    B = torch.cat(edge_feats, dim=-1).detach()
    ret = self.net(B)
    return ret

class MultiClassNodeNetwork(nn.Module):
  def __init__(self, input_dim=2, output_dim=8, nclasses=4,
               hidden_activation=nn.Tanh, how='standard',
               stacks=True, **kwargs):
    super(MultiClassNodeNetwork, self).__init__()
    if not stacks: how = 'standard'
    self.stacks = stacks
    self.nclasses = nclasses
    self.net = nn.Sequential(
      CustomLinear(3*input_dim, output_dim, nclasses, how),
      hidden_activation(),
      CustomLinear(output_dim, output_dim, nclasses, how),
      hidden_activation())

  def forward(self, data):
    row, col = data.edge_index
    mi = data.m.new_zeros(data.m.shape)
    mo = data.m.new_zeros(data.m.shape)
    mi = scatter_add(data.edge_attr*data.m[row], col, dim=0, out=mi)
    mo = scatter_add(data.edge_attr*data.m[col], row, dim=0, out=mo)
    M = torch.cat([mi,mo,data.m],dim=-1)
    return self.net(M)

class GNNDeepMultiHead(nn.Module):
  def __init__(self, input_dim=2, output_dim=4, hidden_dim=8, n_iters=3,
               hidden_activation=nn.Tanh, how='standard',
               stacks=True, edgefeats=False, **kwargs):
    super(GNNDeepMultiHead, self).__init__()
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
      data.edge_attr = self.edge_network(data)
      # mem("after edge network")
      # Apply node network
      H = self.node_network(data)
      # mem("after node network")
      # Shortcut connect the inputs onto the hidden representation
      data.m = torch.cat([H, X], dim=-1)
      # mem(f"after iteration {i}")

    # Apply final edge network
    return self.edge_network(data).squeeze()

