import torch.nn as nn 
import MinkowskiEngine as ME

from .encoder import PanopticEncoder
from .decoder import PanopticDecoder

from Core.activation import minkowski_wrapper

class PanopticNet(nn.Module):
    def __init__(self, n_dims, input_feats, n_classes, n_feats, A, bias, **kwargs):
        nn.Module.__init__(self)
        self.encoder = PanopticEncoder(in_channels=input_feats, A=A, D=n_dims)
        self.semantic_decoder = PanopticDecoder(256,  A=A, D=n_dims)
        self.instance_decoder = PanopticDecoder(256,  A=A, D=n_dims)
       
        self.final_semantic = ME.MinkowskiConvolution(
            in_channels = n_feats, 
            out_channels = n_classes,
            kernel_size=1,
            bias=True,
            dimension= n_dims)

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

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = ME.MinkowskiSigmoid()
        self.dropout = ME.MinkowskiDropout(0.5)

    def forward(self, x):
        out = self.encoder(x)

        out_semantic = self.semantic_decoder(out) 
        out_semantic = self.channelwiseconv(out_semantic)
        out_semantic = self.dropout(out_semantic)
       
        out_instance = self.instance_decoder(out)
        out_instance = self.channelwiseconv(out_instance)
        out_instance = self.dropout(out_instance)
        
        # semantic Head 
        sem = self.final_semantic(out_semantic)
        # Instance Head
        center = self.final_medoid(out_instance)
        offset = self.final_offset(out_instance)
    
        ret  = {'semantic_pred':self.softmax(sem.F), 'center_pred':self.sigmoid(center),'offset_pred':offset} 
        return ret
    
