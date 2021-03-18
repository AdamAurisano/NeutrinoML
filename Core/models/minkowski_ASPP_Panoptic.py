'''Segmentation architecture using MinkowskiEngine UNet'''
from torch.nn import Sequential as Seq, Softmax, Sigmoid
import torch.nn.functional as f
import MinkowskiEngine as ME
from .minkowski_unet import UNetUp, UNetDown
from Core.activation import minkowski_wrapper
from .minkowski_ASPPModule import ASPP

class ASPP_Panoptic(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes, n_feats, rate, A,**kwargs):
    super(ASPP_Panoptic, self).__init__(n_dims)
    
    #input layer
    self.in_net = Seq(
      ME.MinkowskiConvolution(
        in_channels=input_feats,
        out_channels=n_feats,
        kernel_size=3,
        stride=1,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(n_feats),
      minkowski_wrapper(n_dims, A))

    # Downward layers
    self.down_net = UNetDown(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=A)
   

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
      activation=A)

    # Instance Center prediction 
    self.center_pred = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=1,
      kernel_size=1,
      stride=1,
      dimension=n_dims)
    self.sigmoid = Sigmoid()

   # Instance Center Regression 
    self.center_reg = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=3,
      kernel_size=1,
      stride=1,
      dimension=n_dims)
    #self.softplus = Sofplus()

   # Semantic Segmentation Head
    self.semantic_pred = ME.MinkowskiConvolution(
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
    # semantic Head 
    sem = self.semantic_pred(x)
    # Instance Head
    center = self.center_pred(x)
    offset = self.center_reg(x)
    ret  = {'semantic_pred':self.softmax(sem.F), 'center_pred':self.sigmoid(center.F),'offset_pred':offset} 
    return ret
