'''Segmentation architecture using MinkowskiEngine UNet'''
from torch.nn import Sequential as Seq, Softmax
import torch.nn.functional as f
import MinkowskiEngine as ME
from .minkowski_unet import UNetUp, UNetDown
from Core.activation import MinkowskiActivation

class MinkowskiSeg(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes, n_feats, A, **kwargs):
    super(MinkowskiSeg, self).__init__(n_dims)

    self.in_net = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats),
      MinkowskiActivation(n_dims, A))

    # Downward layers
    self.down_net = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=A)
    
    # Upward layers
    self.up_net = UNetUp(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=A)

    self.out_net = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=n_classes,
      kernel_size=1,
      stride=1,
      dimension=n_dims)
    self.softmax = Softmax(dim=1)

  def forward(self, x):

    # Input network
    x = self.in_net(x)
    
    # Down network
    x = self.down_net(x)
    x = self.up_net(x)
    
    x = self.out_net(x)
    return self.softmax(x.F)

