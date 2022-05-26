import torch.nn as nn
import MinkowskiEngine as ME
from Core.activation import minkowski_wrapper


class MinkowskiEnergyNet(ME.MinkowskiNetwork):
    def __init__(self, n_feats, input_feats,  n_dims, bias, **kwargs):
        ME.MinkowskiNetwork.__init__(self,n_dims)
        self.inconv = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = input_feats,
                out_channels = n_feats,
                kernel_size=3,
                bias=bias,
                dimension=n_dims),
            ME.MinkowskiBatchNorm(n_feats),
            ME.MinkowskiReLU(),
        )
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = n_feats,
                out_channels = 32,
                kernel_size=1,
                bias=bias,
                dimension=n_dims),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),
        )
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = n_feats,
                out_channels = 64,
                kernel_size=3,
                bias=bias,
                dimension=n_dims),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels = n_feats,
                out_channels = 128,
                kernel_size=5,
                bias=bias,
                dimension=n_dims),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(),
        )

        self.max_pool = ME.MinkowskiGlobalMaxPooling()

        self.dp = ME.MinkowskiDropout()
        self.linear = ME.MinkowskiLinear(224, 1, bias=bias)

    def forward(self, x: ME.SparseTensor):
        x  = self.inconv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = ME.cat(x1,x2,x3)
        x = self.max_pool(x)
        x = self.dp(x)
        x = self.linear(x)
        return x.F.flatten()
        

