import torch.nn as nn 
import MinkowskiEngine as ME

from .encoder import PanopticEncoder
from .decoder import SemanticDecoderA

from Core.activation import minkowski_wrapper

class SemanticSegNet(nn.Module):
    def __init__(self, n_dims, input_feats, n_classes, n_feats, A, bias, **kwargs):
        nn.Module.__init__(self)
        self.encoder = PanopticEncoder(in_channels=input_feats, A=A, D=n_dims)
        self.semantic_decoder = SemanticDecoderA(256,  A=A, D=n_dims)
       
        self.final_semantic = ME.MinkowskiConvolution(
            in_channels = n_feats, 
            out_channels = n_classes,
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

        self.softmax = nn.Softmax(dim=1)
        self.dropout = ME.MinkowskiDropout(0.5)

    def forward(self, x):
        out = self.encoder(x)
        out = self.semantic_decoder(out) 
        out = self.channelwiseconv(out)
        out = self.dropout(out)
        out = self.final_semantic(out)
    
        return self.softmax(out.F)
    
