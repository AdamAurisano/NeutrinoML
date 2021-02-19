'''Instance Segmentation architecture using MinkowskiEngine UNet'''
from torch.nn import Sequential as Seq, Sigmoid, Softmax
import torch.nn.functional as f
import MinkowskiEngine as ME
from .minkowski_unet import UNetUp, UNetDown
from Core.activation import minkowski_wrapper

class PanopticSeg(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes, n_feats, A, **kwargs):
    super(PanopticSeg, self).__init__(n_dims)

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
    
    # Upward layers
    self.up_net = UNetUp(
      n_dims=n_dims,
      unet_depth=unet_depth,
      n_feats=n_feats,
      activation=A)

    #Semantic prediction
    self.semantic_pred = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=n_classes,
      kernel_size=1,
      stride=1,
      dimension=n_dims)
    self.softmax = Softmax(dim=1)


    # Instance Center prediction 
    self.center_pred = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=1,
      kernel_size=5,
      stride=1,
      dimension=n_dims)
    self.sigmoid = Sigmoid()
   
   # Instance Center Regression 
    self.center_reg = ME.MinkowskiConvolution(
      in_channels=n_feats,
      out_channels=3,
      kernel_size=5,
      stride=1,
      dimension=n_dims)
    #self.sigmoid = Sigmoid()

  def forward(self, x):

    # Input network
    x = self.in_net(x)
    # Down network
    x = self.down_net(x)
    # up network 
    x = self.up_net(x)
    # semantic Head 
    sem = self.semantic_pred(x)
    # Instance Head
    center = self.center_pred(x)
    offset = self.center_reg(x)
    ret  = {'semantic_pred':self.softmax(sem.F), 'center_pred':self.sigmoid(center.F),'offset_pred':offset} 
    return ret
   #return self.softmax(sem.F), self.sigmoid(center.F), offset
    
