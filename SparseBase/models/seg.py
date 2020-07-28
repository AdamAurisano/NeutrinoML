'''
3D sparse segmentation network model
'''
import torch.nn as nn
import torch.nn.functional as F
import sparseconvnet as scn

class SparseSegmentation(nn.Module):
    # to do: figure out spatial extent
    def __init__(self, n_dims=2, extent=6000, unet_depth=5, unet_reps=1, input_feats=1, n_feats=32, res_blocks=False, n_classes=7):
        super(SparseSegmentation, self).__init__()
        planes = [ (i+1)*n_feats for i in range(unet_depth) ]
        self.model = scn.Sequential().add(
            scn.InputLayer(n_dims, extent)).add(
            scn.SubmanifoldConvolution(n_dims, input_feats, n_feats, 3, False)).add(
            scn.UNet(n_dims, unet_reps, planes, residual_blocks=res_blocks, downsample=[2,2])).add(
            scn.BatchNormReLU(n_feats)).add(
            scn.OutputLayer(n_dims))
        self.output = nn.Sequential(nn.Linear(n_feats, n_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return x

