'''Sparse UNet blocks using Minkowski Engine'''
from torch.nn import Sequential as Seq, ModuleList as ML
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

class UNetDown(ME.MinkowskiNetwork):
  def __init__(self, n_dims, unet_depth, n_feats, activation, **kwargs):
    super(UNetDown, self).__init__(n_dims)
    
    self.down = ML()
    for i in range(unet_depth):
      n_in = (2**i) * n_feats
      n_out = (2**(i+1)) * n_feats
      self.down.append(Seq(
        ME.MinkowskiConvolution(
          in_channels=n_in,
          out_channels=n_out,
          kernel_size=3,
          stride=2,
          dimension=n_dims),
        ME.MinkowskiBatchNorm(n_out)))
    
    self.a = activation
    
  def forward(self, x):
    
    x_out = []
    for block in self.down[:-1]:
      x = block(x)
      x_out.append(x) # Skip connections have no activation
      x = self.a(x)

    x = self.down[-1](x)
    x = self.a(x)
    x_out.append(x) # Bottom layer does have activation
    
    return x_out

class UNetUp(ME.MinkowskiNetwork):
  def __init__(self, n_dims, unet_depth, n_feats, activation, **kwargs):
    super(UNetUp, self).__init__(n_dims)
    
    # Upward layers
    self.up = ML()
    for i in range(unet_depth):
      in_feats = 2**(unet_depth-i) * n_feats
      skip_feats = 2**(unet_depth-(i)) * n_feats if i > 0 else 0
      n_in = in_feats + skip_feats
      n_out = 2**(unet_depth-(i+1)) * n_feats if i < unet_depth-1 else n_feats
      self.up.append(Seq(
        ME.MinkowskiConvolutionTranspose(
          in_channels=n_in,
          out_channels=n_out,
          kernel_size=3,
          stride=2,
          dimension=n_dims),
        ME.MinkowskiBatchNorm(n_out)))
    
    self.a = activation
    
  def forward(self, x_skip):
    
    # Start by propagating up bottom layer
    x = x_skip.pop()
    for block, skip in zip(self.up[:-1], reversed(x_skip)):
      x = block(x)
      x = self.a(x)
      x = ME.cat(x, skip)
    
    # Last upward layer
    x = self.up[-1](x)
    x = self.a(x)
    
    return x
