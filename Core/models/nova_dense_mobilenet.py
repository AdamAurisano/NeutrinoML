from torch.nn import Sequential as Seq, Dropout, Linear, ReLU, Softmax, Module, Conv2d, BatchNorm2d, AvgPool2d
import torch 

class Conv(Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1):
        super(Conv, self).__init__()
        self.net = Seq(
            Conv2d(
                in_channels=in_feat,
                out_channels=out_feat,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                padding=int((kernel_size-1)/2)*stride),
            BatchNorm2d(out_feat),
            ReLU(out_feat))
        
    def forward(self, x):
        return self.net(x)

class Bottleneck(Module):
    def __init__(self, in_feat, out_feat, alpha, t, kernel_size, stride, r):
        super(Bottleneck, self).__init__()
        in_feat = int(in_feat * alpha)
        expansion = int(in_feat * t)
        out_feat = int(out_feat * alpha)
        self.r = r
        self.net = Seq(
            Conv( in_feat, expansion, 1, 1),
            Conv2d(
                in_channels=expansion,
                out_channels=expansion,
                kernel_size=kernel_size,
                stride=stride,
                groups=expansion,
                padding=int((kernel_size-1)/2)*stride),
            BatchNorm2d(expansion),
            ReLU(expansion),
            Conv2d(
                in_channels=expansion,
                out_channels=out_feat,
                kernel_size=1,
                stride=1),
            BatchNorm2d(out_feat))
        
    def forward(self, x):
        x_out = self.net(x)
        if self.r: x_out = x_out + x
        return x_out

class InvertedResidual(Module):
    def __init__(self, in_feat, out_feat, alpha, t, kernel_size, stride, n):
        super(InvertedResidual, self).__init__()
        net = [ Bottleneck( in_feat, out_feat, alpha, t, kernel_size, stride, False) ]
        for i in range(1, n):
            net.append(Bottleneck( out_feat, out_feat, alpha, t, kernel_size, 1, True))
        self.net = Seq(*net)
        
    def forward(self, x):
        return self.net(x)

class SubNet(Module):
    def __init__(self, alpha):
        super(SubNet, self).__init__()
        self.net = Seq(
            Conv( 1, int(alpha*32), 3, 2),
            InvertedResidual( 32, 16, alpha, 1, 3, 1, 1),
            InvertedResidual( 16, 24, alpha, 6, 3, 2, 2),
            InvertedResidual( 24, 32, alpha, 6, 3, 2, 3))
        
    def forward(self, x):
        return self.net(x)

class DenseMobileNet(Module):
    def __init__(self, alpha, depth, classes, **kwargs):
        super(DenseMobileNet, self).__init__()
        
        self.input_x = SubNet(alpha)
        self.input_y = SubNet(alpha)
        
        self.net = Seq(
            InvertedResidual( 32, 64, alpha, 6, 3, 2, 4),
            InvertedResidual( 64, 96, alpha, 6, 3, 1, 3),
            InvertedResidual( 96, 160, alpha, 6, 3, 2, 3),
            InvertedResidual( 160, 320, alpha, 6, 3, 1, 1),
            Conv( int(alpha*320), int(alpha*1280)),
            AvgPool2d(kernel_size=[6,5]))
        
        self.final = Seq(
            Dropout(0.4),
            Linear(int(alpha*1280), 1024),
            ReLU(1024),
            Dropout(0.4),
            Linear(1024, classes, bias=False),
            Softmax(dim=1))
        
    def forward(self, x):
        xview = x[0]
        yview = x[1]
        xview = self.input_x(xview)
        yview = self.input_y(yview)
        x = xview + yview
        x = self.net(x).squeeze()
        return self.final(x)

