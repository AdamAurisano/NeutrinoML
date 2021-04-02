import torch.nn as nn
from Core.activation import minkowski_wrapper
import MinkowskiEngine as ME


class Bottleneck(ME.MinkowskiNetwork):
    def __init__(self, D, A, inplanes, planes, stride=1, mode='NORM', k=1, dilation=1):
        """
        Pre-act residual block, the middle transformations are bottle-necked
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        :param mode: NORM | UP
        :param k: times of additive
        """

        super(Bottleneck, self).__init__(D)
        self.mode = mode
        self.relu = minkowski_wrapper(D, A)
        self.k = k

        btnk_ch = planes // 4

        self.bn1 = ME.MinkowskiBatchNorm(inplanes)
        self.conv1 = ME.MinkowskiConvolution(inplanes, btnk_ch, kernel_size=1, bias=False, dimension=D)

        self.bn2 = ME.MinkowskiBatchNorm(btnk_ch)
        self.conv2 = ME.MinkowskiConvolution(btnk_ch, btnk_ch, kernel_size=3, stride=stride,
                               dilation=dilation, bias=False, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(btnk_ch)
        self.conv3 = ME.MinkowskiConvolution(btnk_ch, planes, kernel_size=1, bias=False, dimension=D)

        if mode == 'UP':
            self.shortcut = None
        elif inplanes != planes or stride > 1:
            self.shortcut = nn.Sequential(
                ME.MinkowskiBatchNorm(inplanes),
                self.relu,
                ME.MinkowskiConvolution(inplanes, planes, kernel_size=1, stride=stride, bias=False, dimension=D)
            )
        else:
            self.shortcut = None

    def _pre_act_forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.mode == 'UP':
            residual = self.squeeze_idt(x)
        elif self.shortcut is not None:
            residual = self.shortcut(residual)

#         out += residual

        return out

    def squeeze_idt(self, idt):
        n, c = idt.F.shape
        return idt.F.view(n, c // self.k, self.k).sum(2)

    def forward(self, x):
        out = self._pre_act_forward(x)
        return out
