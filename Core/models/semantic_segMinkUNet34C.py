'''Instance Segmentation architecture using MinkowskiEngine UNet'''
from torch.nn import Sequential as Seq, Sigmoid, Softmax, Softplus
from Core.models.minkunet import MinkUNet34C
from Core.activation import minkowski_wrapper
import MinkowskiEngine as ME

class SemanticSeg(ME.MinkowskiNetwork):

  def __init__(self, n_dims, unet_depth, input_feats, n_classes, n_feats, A, **kwargs):
    super(SemanticSeg, self).__init__(n_dims)
    
    self.unet = MinkUNet34C(input_feats,n_classes)

    self.softmax = Softmax(dim=1)

  def forward(self, x):
    x = self.unet(x)
    return self.softmax(x.F) 
