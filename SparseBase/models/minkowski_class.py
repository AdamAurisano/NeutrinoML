'''Classification architecture using MinkowskiEngine UNet'''
import torch
from torch.nn import Sequential as Seq, Softmax, Linear, ModuleList
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from .minkowski_unet import UNetUp, UNetDown

class MinkowskiClass(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes,
               n_feats, activation=MF.relu, **kwargs):
    super(MinkowskiClass, self).__init__(n_dims)

    self.a = activation

    self.in_net = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats))

    # Downward layers
    self.down_net = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=activation)

    # Global pooling and output
    self.pool = ME.MinkowskiGlobalMaxPooling(dimension=n_dims)
    self.out_net = ME.MinkowskiLinear(n_feats, n_classes, bias=True)
    
  def forward(self, x):

    # Input network
    x = self.in_net(x)
    x = self.a(x)
    
    # Down network
    x = self.down_net(x)
    x = self.up_net(x)
    
    x = self.out_net(x)
    return self.softmax(x.F)



class Minkowski2StackClass(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes,
               n_feats, activation=MF.relu, **kwargs):
    super(Minkowski2StackClass, self).__init__(n_dims)

    self.a = activation

    # Input layers
    self.in_x = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats))
    self.in_y = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats))

    # Downward layers
    self.down_x = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=activation)
    self.down_y = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=activation)

    # Convolution layers
    self.conv_x = ModuleList()
    self.conv_y = ModuleList()
    for i in range(unet_depth):
      hidden = (2**(i+1)) * n_feats
      self.conv_x.append(Seq(
        ME.MinkowskiConvolution(
          in_channels = hidden,
          out_channels = hidden,
          kernel_size=3,
          stride=2,
          dimension=n_dims),
        ME.MinkowskiBatchNorm(hidden)))
      self.conv_y.append(Seq(
        ME.MinkowskiConvolution(
          in_channels = hidden,
          out_channels = hidden,
          kernel_size=3,
          stride=2,
          dimension=n_dims),
        ME.MinkowskiBatchNorm(hidden)))

    # Global pooling and output
    self.pool = ME.MinkowskiGlobalMaxPooling()
    output_size = 0
    for i in range(unet_depth): output_size += 2**(i+2) * n_feats # Add up features from every layer of the UNet
    self.out_net = Seq(
      Linear(output_size, output_size, bias=True),
      Linear(output_size, n_classes, bias=True))

  def forward(self, x):

    # Create sparse tensors
    x1 = ME.SparseTensor(x[0], x[1])
    x2 = ME.SparseTensor(x[2], x[3])

    # Input network
    x1 = self.in_x(x1)
    x1 = self.a(x1)
    x2 = self.in_y(x2)
    x2 = self.a(x2)

    # Down network
    x1 = self.down_x(x1)
    x2 = self.down_y(x2)

    # Concatenate all layers of the UNet
    x1 = [ c(x) for c, x in zip(self.conv_x, x1) ]
    x2 = [ c(x) for c, x in zip(self.conv_y, x2) ]
    x1 = torch.cat([self.pool(x).F for x in x1], dim=-1)
    x2 = torch.cat([self.pool(x).F for x in x2], dim=-1)

    # Concatenate stacks from views and run final linear layer
    x = torch.cat((x1, x2), dim=-1)
    return self.out_net(x)

