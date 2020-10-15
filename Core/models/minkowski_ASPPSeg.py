'''Segmentation architecture using MinkowskiEngine UNet'''
from torch.nn import Sequential as Seq, Softmax
import torch.nn.functional as f
import MinkowskiEngine as ME
from .minkowski_unet import UNetUp, UNetDown
from Core.activation import MinkowskiActivation
from .minkowski_ASPPModule import ASPP

class MinkowskiASPPSeg(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes, n_feats, activation,rate,**kwargs):
    super(MinkowskiASPPSeg, self).__init__(n_dims)
    
    #input layer
    self.in_net = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats),
      MinkowskiActivation(n_dims, activation))

    # Downward layers
    self.down_net = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=activation)
   

    #ASPP Seg Module
    self.aspp = ASPP(
      n_dims = n_dims,
      unet_depth = unet_depth,
      n_feats = n_feats,
      atrous_rates = rate) 

    #Upward layers
    self.up_net = UNetUp(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=activation)

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
    # ASPP Module
    x = self.aspp(x)
    # Up network 
    x = self.up_net(x)
   
    x = self.out_net(x)
    return  self.softmax(x.F)

