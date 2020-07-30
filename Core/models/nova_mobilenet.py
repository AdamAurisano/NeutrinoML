from torch.nn import Sequential as Seq, Dropout, Linear, ReLU, Softmax
import MinkowskiEngine as ME

from torch.nn import ModuleList # delete me

class Conv(ME.MinkowskiNetwork):
    def __init__(self, D, in_feat, out_feat, kernel_size=1, stride=1):
        super(Conv, self).__init__(D)
        self.net = Seq(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=out_feat,
                kernel_size=kernel_size,
                stride=stride,
                has_bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(out_feat),
            ME.MinkowskiReLU(out_feat))
        
    def forward(self, x):
        return self.net(x)

class Bottleneck(ME.MinkowskiNetwork):
    def __init__(self, D, in_feat, out_feat, alpha, t, kernel_size, stride, r):
        super(Bottleneck, self).__init__(D)
        in_feat = int(in_feat * alpha)
        expansion = int(in_feat * t)
        out_feat = int(out_feat * alpha)
        self.r = r
        self.net = Seq(
            Conv(D, in_feat, expansion, 1, 1),
            ME.MinkowskiChannelwiseConvolution(
                in_channels=expansion,
                kernel_size=kernel_size,
                stride=stride,
                dimension=D),
            ME.MinkowskiBatchNorm(expansion),
            ME.MinkowskiReLU(expansion),
            ME.MinkowskiConvolution(
                in_channels=expansion,
                out_channels=out_feat,
                kernel_size=1,
                stride=1,
                dimension=D),
            ME.MinkowskiBatchNorm(out_feat))
        
    def forward(self, x):
        x_out = self.net(x)
        if self.r: x_out = x_out + x
        return x_out

class InvertedResidual(ME.MinkowskiNetwork):
    def __init__(self, D, in_feat, out_feat, alpha, t, kernel_size, stride, n):
        super(InvertedResidual, self).__init__(D)
        net = [ Bottleneck(D, in_feat, out_feat, alpha, t, kernel_size, stride, False) ]
        for i in range(1, n):
            net.append(Bottleneck(D, out_feat, out_feat, alpha, t, kernel_size, 1, True))
        self.net = Seq(*net)
        
    def forward(self, x):
        return self.net(x)

class SubNet(ME.MinkowskiNetwork):
    def __init__(self, D, alpha):
        super(SubNet, self).__init__(D)
        self.net = Seq(
            Conv(D, 1, int(alpha*32), 3, 2),
            InvertedResidual(D, 32, 16, alpha, 1, 3, 1, 1),
            InvertedResidual(D, 16, 24, alpha, 6, 3, 2, 2),
            InvertedResidual(D, 24, 32, alpha, 6, 3, 2, 3))
        
    def forward(self, x):
        return self.net(x)

class MobileNet(ME.MinkowskiNetwork):
    def __init__(self, D, alpha, depth, classes, **kwargs):
        super(MobileNet, self).__init__(D)
        
        self.input_x = SubNet(D, alpha)
        self.input_y = SubNet(D, alpha)
        
        self.merge = ME.MinkowskiUnion()
        
        self.net = Seq(
            InvertedResidual(D, 32, 64, alpha, 6, 3, 2, 4),
            InvertedResidual(D, 64, 96, alpha, 6, 3, 1, 3),
            InvertedResidual(D, 96, 160, alpha, 6, 3, 2, 3),
            InvertedResidual(D, 160, 320, alpha, 6, 3, 1, 1),
            Conv(D, int(alpha*320), int(alpha*1280)),
            ME.MinkowskiGlobalPooling())
        
        self.final = Seq(
            Dropout(0.4),
            Linear(int(alpha*1280), 1024),
            ReLU(1024),
            Dropout(0.4),
            Linear(1024, classes, bias=False),
            Softmax(dim=1))
        
    def forward(self, x):
        xview = ME.SparseTensor(x[0], x[1])
        yview = ME.SparseTensor(x[2], x[3])
        x = self.merge(self.input_x(xview), self.input_y(xview))
        x = self.net(x)
        return self.final(x.F)
        