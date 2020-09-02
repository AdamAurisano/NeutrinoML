'''ASPP module using Minkowski Engine'''
from torch.nn import Sequential as Seq, ModuleList as ML
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
from Core.activation import MinkowskiActivation
import torch.nn as nn

class ASPPConvMinkowski(ME.MinkowskiNetwork):
  def __init__(self,n_dims, input_feats, out_feats, dilation):
    super(ASPPConvMinkowski, self).__init__(n_dims)
    self.asppconv = Seq(
      ME.MinkowskiConvolution(
        in_channels = input_feats,
        out_channels = out_feats,
        kernel_size=3,
        dilation=dilation,
        has_bias=False,
        dimension=n_dims),
        ME.MinkowskiBatchNorm(out_feats),
        MinkowskiActivation(n_dims,nn.ReLU())
    )

  def forward(self,x):
    x = self.asppconv(x)
    return x

class ASPPPoling(ME.MinkowskiNetwork):
  def __init__(self, n_dims,input_feats):
    super(ASPPPoling,self).__init__(n_dims)
    self.aspppoling = Seq(
      ME.MinkowskiSumPooling(
        kernel_size = 1,
        stride=8,
        dilation=1,
        dimension=n_dims),
      ME.MinkowskiConvolution(
        in_channels = input_feats,
        out_channels = input_feats,
        kernel_size=1,
        has_bias=False,
        dimension=n_dims),
        ME.MinkowskiBatchNorm(input_feats),
        MinkowskiActivation(n_dims,nn.ReLU()),
     ME.MinkowskiPoolingTranspose(
       kernel_size =1,
       stride=8,
       dilation=1,
       kernel_generator=None,
       dimension=n_dims
     )
    )
  def forward(self,x):
    x = self.aspppoling(x)
    return x



class ASPP(ME.MinkowskiNetwork):
  def __init__(self,n_dims, unet_depth, n_feats, atrous_rates):
    super(ASPP, self).__init__(n_dims)
    input_feats = 2**(unet_depth) * n_feats 
    self.mod = ML()
    ##self.aspp = Seq(
    self.mod.append(Seq(
      ME.MinkowskiConvolution(
        in_channels = input_feats,
        out_channels = input_feats,
        kernel_size=1,
        has_bias=False,
        dimension=n_dims),
      ME.MinkowskiBatchNorm(input_feats),
      MinkowskiActivation(n_dims,nn.ReLU())
    ))
    rates = tuple(atrous_rates)
    for rate in rates:
      self.mod.append(ASPPConvMinkowski(
              n_dims = n_dims,
              input_feats = input_feats,
              out_feats = input_feats,
              dilation = rate))
    self.mod.append(ASPPPoling(n_dims, input_feats))
    n = len(rates)
    self.lastConv = Seq(
      ME.MinkowskiConvolution(
        in_channels = (n+2)*input_feats,
        out_channels = input_feats,
        kernel_size =1,
        has_bias = False,
        dimension = n_dims),
      ME.MinkowskiBatchNorm(input_feats),
      MinkowskiActivation(n_dims,nn.ReLU())
    )
  def forward(self,xlist):
    x = xlist.pop()
    res = []
    for conv in self.mod:
        x = conv(x)
        res.append(conv(x))
    x_cat = ME.cat(res[0],res[1],res[2],res[3],res[4])
    x_cat = self.lastConv(x_cat) 
    xlist.append(x_cat)
    return xlist 
      

