import MinkowskiEngine as ME
import torch.nn as nn

from Core.models.encoder import EncoderBase
from Core.activation import minkowski_wrapper
from Core.models.resnet_block import BasicBlock, Bottleneck

class DecoderBase(EncoderBase):
    BLOCK = None
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)
    INIT_DIM = 32
    INIT_DIM_DEC = 256
    
    
    def __init__(self, in_channels, A, D =3):
        '''in_chanels = number of chanels from the last plane in the encoder '''
        nn.Module.__init__(self)
        self.D = D
        self.network_initialization(A, D)
        
    def network_initialization(self, A, D):
        self.inplanes = self.INIT_DIM_DEC
        self.activation = minkowski_wrapper(D, A) 
        
        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4],activation=A)
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5],activation=A)
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6],activation=A)
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7],activation=A)
    def forward(self,in_dict):
        
        input_layer = in_dict['input']
        out_b1 = in_dict['b1']
        out_b2 = in_dict['b2']
        out_b3 = in_dict['b3']
        out_b4 = in_dict['b4']
        
        # tensor_stride=8
        out = self.convtr4p16s2(out_b4)
        out = self.bntr4(out)
        out = self.activation(out)

        out = ME.cat(out, out_b3)
        out = self.block5(out)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.activation(out)

        out = ME.cat(out, out_b2)
        out = self.block6(out)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.activation(out)

        out = ME.cat(out, out_b1)
        out = self.block7(out)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.activation(out)

        out = ME.cat(out, input_layer)
        out = self.block8(out)
        return out

class PanopticDecoder(DecoderBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)
class InstanceDecoder(DecoderBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 128, 32)
