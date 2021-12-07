"""
This module implements the PyTorch modules that define the
message-passing graph neural networks for hit or segment classification.
"""

import torch, torch.cuda as tc
import torch.nn as nn
from torch_scatter import scatter_add
import pandas as pd, torch, torch_geometric as tg
import numpy as np

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
    plane='u',
    **kwargs,
  ):

    super(MultiClassEdgeNetwork, self).__init__()
    self.plane = plane
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

  def forward(self, data):
    row,col = getattr(data, "edge_index_" + self.plane)
    m = getattr(data, "m_" + self.plane)
    edge_feats = [ m[col], m[row] ]
    B = torch.cat(edge_feats, dim=-1).detach()
    ret = self.net(B)
    return ret

class MultiClassNodeNetwork(nn.Module):
  def __init__(self, input_dim=2, output_dim=8, nclasses=4,
               hidden_activation=nn.Tanh, how='standard',
               stacks=True, plane='u', **kwargs):
    super(MultiClassNodeNetwork, self).__init__()
    self.plane = plane
    if not stacks: how = 'standard'
    self.stacks = stacks
    self.nclasses = nclasses
    self.net = nn.Sequential(
      CustomLinear(3*input_dim, output_dim, nclasses, how),
      hidden_activation(),
      CustomLinear(output_dim, output_dim, nclasses, how),
      hidden_activation())

  def forward(self, data):
    row,col = getattr(data, "edge_index_" + self.plane)                         
    m = getattr(data, "m_" + self.plane) 
    mi = m.new_zeros(m.shape)
    mo = m.new_zeros(m.shape)
    edge_attr = getattr(data, "edge_attr_" + self.plane)
    mi = scatter_add(edge_attr*m[row], col, dim=0, out=mi)
    mo = scatter_add(edge_attr*m[col], row, dim=0, out=mo)
    M = torch.cat([mi,mo,m],dim=-1)
    return self.net(M)

def _get_plane(i):                                                            
  if i == 0: plane = 'u'                                                      
  elif i == 1: plane = 'v'                                                    
  else: plane = 'y'                                                           
  return plane  

class SpacePointNetwork(nn.Module):                                         
  def __init__(self, input_dim=2, output_dim=8, nclasses=4,                     
               hidden_activation=nn.Tanh, how='standard',
               **kwargs):                                          
    super(SpacePointNetwork, self).__init__()
    how = 'standard'
    self.nclasses = nclasses                                                    
    self.hidden_feats = input_dim
    self.net = nn.Sequential(                                                   
      CustomLinear(3*input_dim, output_dim, nclasses, how),                     
      hidden_activation(),                                                      
      CustomLinear(output_dim, output_dim, nclasses, how),                      
      hidden_activation())                                                      
                                                                                
  def forward(self, data):                          
    m_sp = [None] * 3
    for i in range(3):
      row, col = getattr(data, "edge_index_3d_" + _get_plane(i))
      m_sp[i] = data.H[i].new_zeros(sum(data.n_sp), self.nclasses, self.hidden_feats)
      m_sp[i] = scatter_add(data.H[i][row], col, dim=0, out=m_sp[i])

    M = torch.cat(m_sp, dim=-1)                                      
    M = self.net(M)

    H = [None] * 3
    for i in range(3):
      row, col = getattr(data, "edge_index_3d_" + _get_plane(i))   
      H[i] = data.H[i].new_zeros(data.H[i].shape)
      H[i] = scatter_add(M[col], row, dim=0, out=H[i])

      # print("down shape on " + str(i))
      # print(H[i].shape)

    return H

class GNN3DeepMultiHead(nn.Module): 

  def __init__(self, input_dim=2, output_dim=4, hidden_dim=8, n_iters=3,
               hidden_activation=nn.Tanh, how='standard',
               stacks=True, edgefeats=False, classify_node=False, 
               use_spacepoints=False, **kwargs):
    super(GNN3DeepMultiHead, self).__init__()
    self.n_iters = n_iters
    self.nclasses = output_dim
    self.stacks = stacks
    self.classify_node = classify_node
    self.use_spacepoints = use_spacepoints
    if not stacks: how = 'standard'
    
    self.input_networks = [None] * 3
    self.edge_networks = [None] * 3
    self.node_networks = [None] * 3

    # node output networks
    if classify_node:
        self.output_networks = [None] * 3

    # setting up per plane
    for i in range(3):
      plane = _get_plane(i)

      self.input_networks[i] = nn.Sequential(                                      
        CustomLinear(input_dim, hidden_dim, output_dim, how),                     
        hidden_activation(),                                                      
      )

      self.edge_networks[i] = MultiClassEdgeNetwork(                               
        input_dim + hidden_dim,                                                   
        hidden_dim,                                                               
        output_dim,                                                               
        hidden_activation,                                                        
        how,                                                                      
        stacks,                                                                   
        edgefeats,
        plane,                                                                
       )

      self.node_networks[i] = MultiClassNodeNetwork(                               
        input_dim + hidden_dim,                                                   
        hidden_dim,                                                               
        output_dim,                                                               
        hidden_activation,                                                        
        how,
        stacks,
        plane,                                                                      
      )
    
      if classify_node:
        output = 1 if stacks else self.nclasses
        self.output_networks[i] = nn.Sequential(                                    
          CustomLinear(hidden_dim, output, output_dim, how),                      
          hidden_activation(),                                                    
        )     
   
    self.input_networks = nn.ModuleList(self.input_networks)
    self.node_networks = nn.ModuleList(self.node_networks)
    self.edge_networks = nn.ModuleList(self.edge_networks)

    if classify_node:                 
      self.output_networks = nn.ModuleList(self.output_networks)

    if use_spacepoints:
      self.spacepoint_network = SpacePointNetwork(
        hidden_dim,
        hidden_dim,
        output_dim,
        hidden_activation,
        how,
      )

  def forward(self, data):
    """Apply forward pass of the model"""
    global dev
    dev = data.x_u.device

    X = [None] * 3
    data.H = [None] * 3
    for i in range(3):
      plane = _get_plane(i)
      
      X[i] = getattr(data, "x_" + plane).unsqueeze(1).repeat(1, self.nclasses, 1) if self.stacks else getattr(data, "x_" + plane).clone()

      data.H[i] = self.input_networks[i](X[i])

      # Shortcut connect the inputs onto the hidden representation
      setattr(data, "m_" + plane, torch.cat([data.H[i], X[i]], dim=-1))

      # print("m shape on plane" + str(i))
      # print(getattr(data, "m_" + plane).shape)
    
    for i in range(self.n_iters):
      for j in range(3):

        # Apply edge network, update edge_attrs
        setattr(data, "edge_attr_" + _get_plane(j), self.edge_networks[j](data))

        # print("edge shapes on plane " + str(j))
        # print(getattr(data, "edge_attr_" + _get_plane(j)).shape)
        
        # Apply node network
        data.H[j] = self.node_networks[j](data)

        # print("hidden node shapes on plane " + str(j))                                 
        # print(data.H[j].shape)        
 
      # Apply spacepoint network
      if self.use_spacepoints:
        data.H = self.spacepoint_network(data)     
 
      # Shortcut connect the inputs onto the hidden representation
      for j in range(3):  
        setattr(data, "m_" + _get_plane(j), torch.cat([data.H[j], X[j]], dim=-1))              
      
    # Apply final output networks
    ret = [None] * 3
    if self.classify_node:
      for i in range(3): ret[i] = self.output_networks[i](data.H[i]).squeeze()
    else:
      for i in range(3): ret[i] = self.edge_networks[i](data).squeeze()

    return torch.cat(ret, dim=0)

