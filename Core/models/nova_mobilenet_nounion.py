import torch
from torch.nn import Sequential as Seq, Dropout, Linear, ReLU, Softmax
import MinkowskiEngine as ME
from .nova_mobilenet import Conv, InvertedResidual, SubNet    

class MobileNetNoUnion(ME.MinkowskiNetwork):
    def __init__(self, D, A, alpha, depth, classes, dropout, **kwargs):
        super(MobileNetNoUnion, self).__init__(D)

        self.input_x = SubNet(D, A, alpha)
        self.input_y = SubNet(D, A, alpha)
        
        self.net_x = Seq(
            InvertedResidual(D, A, 32, 64, alpha, 6, 3, 2, 4),
            InvertedResidual(D, A, 64, 96, alpha, 6, 3, 1, 3),
            InvertedResidual(D, A, 96, 160, alpha, 6, 3, 2, 3),
            InvertedResidual(D, A, 160, 320, alpha, 6, 3, 1, 1),
            Conv(D, A, int(alpha*320), int(alpha*1280)),
            ME.MinkowskiGlobalPooling())
        
        self.net_y = Seq(
            InvertedResidual(D, A, 32, 64, alpha, 6, 3, 2, 4),
            InvertedResidual(D, A, 64, 96, alpha, 6, 3, 1, 3),
            InvertedResidual(D, A, 96, 160, alpha, 6, 3, 2, 3),
            InvertedResidual(D, A, 160, 320, alpha, 6, 3, 1, 1),
            Conv(D, A, int(alpha*320), int(alpha*1280)),
            ME.MinkowskiGlobalPooling())
        
        self.final = Seq(
            Dropout(dropout),
            Linear(int(alpha*2560), 1024),
            A,
            Dropout(dropout),
            Linear(1024, classes, bias=False))
                
    def forward(self, x):
        xview = ME.SparseTensor(x[0], x[1])
        yview = ME.SparseTensor(x[2], x[3])
        
        xview = self.input_x(xview)
        yview = self.input_y(yview)
        
        xview = self.net_x(xview)
        yview = self.net_y(yview)
        
        x = torch.cat([xview.F, yview.F], dim=1)
        return self.final(x)
        

