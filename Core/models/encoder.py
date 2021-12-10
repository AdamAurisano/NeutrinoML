import torch
import torch.nn as nn
import MinkowskiEngine as ME

from .resnet_block import BasicBlock, Bottleneck
from Core.activation import minkowski_wrapper

class EncoderBase(nn.Module):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2)
    PLANES = (32, 64, 128, 256) 
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    def __init__(self, in_channels, A, D=3):
        self.D = D
        nn.Module.__init__(self) 
        self.network_initialization(in_channels, A, D)
        #self.weight_initialization()

    def network_initialization(self, in_channels, A, D):
        self.inplanes = self.INIT_DIM
        self.activation = minkowski_wrapper(D, A)

        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0],activation=A)

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1],activation=A)

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2],activation=A)

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3],activation=A)

    def _make_layer(self, block, planes, blocks, activation, stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=self.D,
                ),
                ME.MinkowskiBatchNorm(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                activation,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                dimension=self.D,
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, activation, stride=1, dilation=dilation, dimension=self.D
                )
            )

        return nn.Sequential(*layers)

    def weight_initialization(self):
        for m in self.modules():
            print(m)
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)


    def forward(self, x):

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.activation(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.activation(out) 
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.activation(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.activation(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.activation(out)
        out_b4p16 = self.block4(out)


        return {'input':out_p1, 'b1':out_b1p2, 'b2':out_b2p4, 'b3':out_b3p8, 'b4':out_b4p16} 

class PanopticEncoder(EncoderBase):
    BLOCK = BasicBlock
    Planes = (32, 64, 128, 256)
    LAYERS = (2, 3, 4, 6)
class InstanceEncoder(EncoderBase):
    BLOCK = BasicBlock
    Planes = (32, 64, 128, 256)
    LAYERS = (2, 3, 4, 6)
class SemanticEncoder(EncoderBase):
    BLOCK = BasicBlock
    Planes = (32, 64, 128, 256)
    LAYERS = (2, 3, 4, 6)
class SemanticEncoderA(EncoderBase):
    BLOCK = BasicBlock
    Planes = (32, 64, 128, 256)
    LAYERS = (2, 4, 6, 8)
class SemanticEncoderB(EncoderBase):
    BLOCK = BasicBlock
    Planes = (32, 64, 128, 256)
    LAYERS = (2, 3, 4, 5)
