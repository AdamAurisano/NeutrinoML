import torch.nn as nn 
import MinkowskiEngine as ME

from .encoder import InstanceEncoder
from .decoder import PanopticDecoder, InstanceDecoder

from Core.activation import minkowski_wrapper

class InstanceSegNet(nn.Module):
    def __init__(self, n_dims, input_feats, n_feats, A, bias, **kwargs):
        nn.Module.__init__(self)
        self.encoder = InstanceEncoder(in_channels=input_feats, A=A, D=n_dims)
        #self.instance_decoder = PanopticDecoder(256,  A=A, D=n_dims)
        self.instance_decoder = InstanceDecoder(256,  A=A, D=n_dims)
      
        self.final_medoid = ME.MinkowskiConvolution(
            in_channels = n_feats, 
            out_channels = 1, 
            kernel_size=1,
            bias=True,
            dimension= n_dims)

        self.final_offset = ME.MinkowskiConvolution(
            in_channels = n_feats, 
            out_channels = 3,
            kernel_size=1,
            bias=True,
            dimension= n_dims)
        
        self.channelwiseconv = ME.MinkowskiChannelwiseConvolution(
            in_channels = n_feats,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=bias,
            dimension=n_dims)


        self.sigmoid = ME.MinkowskiSigmoid()
        self.softmax = ME.MinkowskiSoftmax(dim=0)
        self.dropout = ME.MinkowskiDropout(0.5)

    def forward(self, x):
        out = self.encoder(x)
        out = self.instance_decoder(out)
        out =  self.channelwiseconv(out)
        out = self.dropout(out)
        # Instance Head
        center = self.final_medoid(out)
        offset = self.final_offset(out)
    

        ret  = {'center_pred':self.sigmoid(center),'offset_pred':offset} 
        return ret
    
