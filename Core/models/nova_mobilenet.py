import torch
from torch.nn import Sequential as Seq, Dropout, Linear, ReLU, Softmax
from Core.activation import MinkowskiActivation
import MinkowskiEngine as ME

class Conv(ME.MinkowskiNetwork):
    def __init__(self, D, A, in_feat, out_feat, kernel_size=1, stride=1):
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
            MinkowskiActivation(D, A))
        
    def forward(self, x):
        return self.net(x)

class Bottleneck(ME.MinkowskiNetwork):
    def __init__(self, D, A, in_feat, out_feat, alpha, t, kernel_size, stride, r):
        super(Bottleneck, self).__init__(D)
        in_feat = int(in_feat * alpha)
        expansion = int(in_feat * t)
        out_feat = int(out_feat * alpha)
        self.r = r
        self.net = Seq(
            Conv(D, A, in_feat, expansion, 1, 1),
            ME.MinkowskiChannelwiseConvolution(
                in_channels=expansion,
                kernel_size=kernel_size,
                stride=stride,
                dimension=D),
            ME.MinkowskiBatchNorm(expansion),
            MinkowskiActivation(D, A),
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
    def __init__(self, D, A, in_feat, out_feat, alpha, t, kernel_size, stride, n):
        super(InvertedResidual, self).__init__(D)
        net = [ Bottleneck(D, A, in_feat, out_feat, alpha, t, kernel_size, stride, False) ]
        for i in range(1, n):
            net.append(Bottleneck(D, A, out_feat, out_feat, alpha, t, kernel_size, 1, True))
        self.net = Seq(*net)
        
    def forward(self, x):
        return self.net(x)

class SubNet(ME.MinkowskiNetwork):
    def __init__(self, D, A, alpha):
        super(SubNet, self).__init__(D)
        self.net = Seq(
            Conv(D, A, 1, int(alpha*32), 3, 2),
            InvertedResidual(D, A, 32, 16, alpha, 1, 3, 1, 1),
            InvertedResidual(D, A, 16, 24, alpha, 6, 3, 2, 2),
            InvertedResidual(D, A, 24, 32, alpha, 6, 3, 2, 3))
        
    def forward(self, x):
        return self.net(x)

class MobileNet(ME.MinkowskiNetwork):
    def __init__(self, D, A, alpha, depth, classes, dropout, **kwargs):
        super(MobileNet, self).__init__(D)

        self.input_x = SubNet(D, A, alpha)
        self.input_y = SubNet(D, A, alpha)
        
        self.net = Seq(
            InvertedResidual(D, A, 32, 64, alpha, 6, 3, 2, 4),
            InvertedResidual(D, A, 64, 96, alpha, 6, 3, 1, 3),
            InvertedResidual(D, A, 96, 160, alpha, 6, 3, 2, 3),
            InvertedResidual(D, A, 160, 320, alpha, 6, 3, 1, 1),
            Conv(D, A, int(alpha*320), int(alpha*1280)),
            ME.MinkowskiGlobalPooling())
        
        self.final = Seq(
            Dropout(dropout),
            Linear(int(alpha*1280), 1024),
            A,
            Dropout(dropout),
            Linear(1024, classes, bias=False))
        
    def merge_views(self, x, y):
        
        x = x.sparse()[0]
        y = y.sparse()[0]
        i = torch.cat([x._indices(), y._indices()], dim=1)
        v = torch.cat([x._values(), y._values()], dim=0)
        
        sparse = torch.sparse.FloatTensor(i, v).coalesce()
        f = sparse._values()
        c = sparse._indices().float().transpose(0,1)
        return ME.SparseTensor(f, c)
        
    def forward(self, x):
        
        xview = ME.SparseTensor(x[0], x[1])
        yview = ME.SparseTensor(x[2], x[3])
        
        xview = self.input_x(xview)
        yview = self.input_y(yview)
        
        x = self.merge_views(xview, yview)
        x = self.net(x)
        return self.final(x.F)
        
