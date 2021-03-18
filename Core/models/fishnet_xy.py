'''
FishNet
Author: Shuyang Sun
'''
from __future__ import division
import torch
import math
from .fish_block import *
from Core.activation import minkowski_wrapper
import MinkowskiEngine as ME

__all__ = ['fish']


class FishBody(ME.MinkowskiNetwork):
    def __init__(self, block, D, A, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None, **kwargs):
        super(FishBody, self).__init__(D)
        self.block = block
        self.trans_map = trans_map
        self.upsample = ME.MinkowskiConvolutionTranspose
        self.down_sample = ME.MinkowskiMaxPooling(2, stride=2, dimension=D)
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.network_planes = network_planes[1:]
        self.depth = len(self.network_planes)
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks
        self.body = self._make_body(D, A, network_planes[0])

    def _make_score(self, D, A, in_ch, out_ch=1000, has_pool=False):
        bn = ME.MinkowskiBatchNorm(in_ch)
        relu = minkowski_wrapper(D, A)
        conv_trans = ME.MinkowskiConvolution(in_ch, in_ch // 2, kernel_size=1, bias=False, dimension=D)
        bn_out = ME.MinkowskiBatchNorm(in_ch // 2)
        conv = nn.Sequential(bn, relu, conv_trans, bn_out, relu)
        if has_pool:
            fc = nn.Sequential(
                ME.MinkowskiGlobalPooling(),
                ME.MinkowskiConvolution(in_ch // 2, out_ch, kernel_size=1, bias=True, dimension=D))
        else:
            fc = ME.MinkowskiConvolution(in_ch // 2, out_ch, kernel_size=1, bias=True, dimension=D)
        return [conv, fc]

    def _make_se_block(self, D, A, in_ch, out_ch):
        bn = ME.MinkowskiBatchNorm(in_ch)
        sq_conv = ME.MinkowskiConvolution(in_ch, out_ch // 16, kernel_size=1, dimension=D)
        ex_conv = ME.MinkowskiConvolution(out_ch // 16, out_ch, kernel_size=1, dimension=D)
        return nn.Sequential(bn,
                             minkowski_wrapper(D, A),
                             ME.MinkowskiGlobalPooling(),
                             sq_conv,
                             minkowski_wrapper(D, A),
                             ex_conv,
                             nn.Sigmoid())

    def _make_residual_block(self, D, A, inplanes, outplanes, nstage, is_up=False, k=1, dilation=1):
        layers = []

        if is_up:
            layers.append(self.block(D, A, inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.append(self.block(D, A, inplanes, outplanes, stride=1))
        for i in range(1, nstage):
            layers.append(self.block(D, A, outplanes, outplanes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_stage(self, D, A, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):
        sample_block = []
        if has_score:
            sample_block.extend(self._make_score(D, A, outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(D, A, inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(D, A, inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)

        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.append(self._make_residual_block(D, A, trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling and is_down_sample:
            sample_block.append(self.down_sample)
        elif not no_sampling:  # Up-Sample
            sample_block.append(self.upsample(in_channels=outplanes, out_channels=outplanes, kernel_size=2, stride=2, dimension=D))

        return nn.ModuleList(sample_block)

    def _make_body(self, D, A, in_planes):
        def get_trans_planes(index):
            map_id = self.trans_map[index-self.num_down-1] - 1
            p = in_planes if map_id == -1 else cated_planes[map_id]
            return p

        def get_trans_blk(index):
            return self.num_trans_blks[index-self.num_down-1]

        def get_cur_planes(index):
            return self.network_planes[index]

        def get_blk_num(index):
            return self.num_res_blks[index]

        cated_planes, fish = [in_planes] * self.depth, []
        for i in range(self.depth):
            # even num for down-sample, odd for up-sample
            is_down, has_trans, no_sampling = i not in range(self.num_down, self.num_down+self.num_up+1),\
                i > self.num_down, i == self.num_down
            cur_planes, trans_planes, cur_blocks, num_trans =\
                get_cur_planes(i), get_trans_planes(i), get_blk_num(i), get_trans_blk(i)

            stg_args = [is_down, cated_planes[i - 1], cur_planes, cur_blocks]

            if is_down or no_sampling:
                k, dilation = 1, 1
            else:
                k, dilation = cated_planes[i - 1] // cur_planes, 2 ** (i-self.num_down-1)

            sample_block = self._make_stage(D, A, *stg_args, has_trans=has_trans, trans_planes=trans_planes,
                                            has_score=(i==self.num_down), num_trans=num_trans, k=k, dilation=dilation,
                                            no_sampling=no_sampling)
            if i == self.depth - 1:
                sample_block.extend(self._make_score(D, A, cur_planes + trans_planes, out_ch=self.num_cls, has_pool=True))
            elif i == self.num_down:
                sample_block.append(nn.Sequential(self._make_se_block(D, A, cur_planes*2, cur_planes)))

            if i == self.num_down-1:
                cated_planes[i] = cur_planes * 2
            elif has_trans:
                cated_planes[i] = cur_planes + trans_planes
            else:
                cated_planes[i] = cur_planes
            fish.append(sample_block)
        return nn.ModuleList(fish[:7])
    
    def _fish_forward(self, all_feat):
        def _concat(a, b):
            return torch.cat([a, b], dim=1)

        def stage_factory(*blks):
            def stage_forward(*inputs):
                if stg_id < self.num_down:  # tail
                    tail_blk = nn.Sequential(*blks[:2])
                    return tail_blk(*inputs)
                elif stg_id == self.num_down:
                    score_blks = nn.Sequential(*blks[:2])
                    score_feat = score_blks(inputs[0])
                    att_feat = blks[3](score_feat)
                    return blks[2](score_feat) * att_feat + att_feat
                else:  # refine
                    feat_trunk = blks[2](blks[0](inputs[0]))
                    feat_branch = blks[1](inputs[1])
                return _concat(feat_trunk, feat_branch)
            return stage_forward

        stg_id = 0
        # tail:
        while stg_id < self.depth:
            #stg_blk = stage_factory(*self.fish[stg_id])
            stg_blk_body = stage_factory(*self.body[stg_id])
            if stg_id <= self.num_down:
                in_feat = [all_feat[stg_id]]
            else:
                trans_id = self.trans_map[stg_id-self.num_down-1]
                in_feat = [all_feat[stg_id], all_feat[trans_id]]

            all_feat[stg_id + 1] = stg_blk_body(*in_feat)
            stg_id += 1
            # loop exit
            if stg_id == self.depth:
                #score_feat = self.fish[self.depth-1][-2](all_feat[-1])
                #score = self.fish[self.depth-1][-1](score_feat)
                score_feat = self.body[self.depth-1][-2](all_feat[-1])
                score = self.body[self.depth-1][-1](score_feat)
                return score

    def forward(self, x):
        all_feat = [None] * (self.depth + 1)
        all_feat[0] = x
        return self._fish_forward(all_feat)


class FishHead(ME.MinkowskiNetwork):
    def __init__(self, block, D, A, num_cls=1000, num_down_sample=5, num_up_sample=3, trans_map=(2, 1, 0, 6, 5, 4),
                 network_planes=None, num_res_blks=None, num_trans_blks=None, **kwargs):
        super(FishHead, self).__init__(D)
        self.block = block
        self.trans_map = trans_map
        self.upsample = ME.MinkowskiConvolutionTranspose
        self.down_sample = ME.MinkowskiMaxPooling(2, stride=2, dimension=D)
        self.num_cls = num_cls
        self.num_down = num_down_sample
        self.num_up = num_up_sample
        self.network_planes = network_planes[1:]
        self.depth = len(self.network_planes)
        self.num_trans_blks = num_trans_blks
        self.num_res_blks = num_res_blks
        self.head = self._make_head(D, A, network_planes[0])

    def _make_score(self, D, A, in_ch, out_ch=1000, has_pool=False):
        bn = ME.MinkowskiBatchNorm(in_ch)
        relu = minkowski_wrapper(D, A)
        conv_trans = ME.MinkowskiConvolution(in_ch, in_ch // 2, kernel_size=1, bias=False, dimension=D)
        bn_out = ME.MinkowskiBatchNorm(in_ch // 2)
        conv = nn.Sequential(bn, relu, conv_trans, bn_out, relu)
        if has_pool:
            fc = nn.Sequential(
                ME.MinkowskiGlobalPooling(),
                ME.MinkowskiConvolution(in_ch // 2, out_ch, kernel_size=1, bias=True, dimension=D))
        else:
            fc = ME.MinkowskiConvolution(in_ch // 2, out_ch, kernel_size=1, bias=True, dimension=D)
        return [conv, fc]

    def _make_se_block(self, D, A, in_ch, out_ch):
        bn = ME.MinkowskiBatchNorm(in_ch)
        sq_conv = ME.MinkowskiConvolution(in_ch, out_ch // 16, kernel_size=1, dimension=D)
        ex_conv = ME.MinkowskiConvolution(out_ch // 16, out_ch, kernel_size=1, dimension=D)
        return nn.Sequential(bn,
                             minkowski_wrapper(D, A),
                             ME.MinkowskiGlobalPooling(),
                             sq_conv,
                             minkowski_wrapper(D, A),
                             ex_conv,
                             nn.Sigmoid())

    def _make_residual_block(self, D, A, inplanes, outplanes, nstage, is_up=False, k=1, dilation=1):
        layers = []

        if is_up:
            layers.append(self.block(D, A, inplanes, outplanes, mode='UP', dilation=dilation, k=k))
        else:
            layers.append(self.block(D, A, inplanes, outplanes, stride=1))
        for i in range(1, nstage):
            layers.append(self.block(D, A, outplanes, outplanes, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def _make_stage(self, D, A, is_down_sample, inplanes, outplanes, n_blk, has_trans=True,
                    has_score=False, trans_planes=0, no_sampling=False, num_trans=2, **kwargs):
        sample_block = []
        if has_score:
            sample_block.extend(self._make_score(D, A, outplanes, outplanes * 2, has_pool=False))

        if no_sampling or is_down_sample:
            res_block = self._make_residual_block(D, A, inplanes, outplanes, n_blk, **kwargs)
        else:
            res_block = self._make_residual_block(D, A, inplanes, outplanes, n_blk, is_up=True, **kwargs)

        sample_block.append(res_block)

        if has_trans:
            trans_in_planes = self.in_planes if trans_planes == 0 else trans_planes
            sample_block.append(self._make_residual_block(D, A, trans_in_planes, trans_in_planes, num_trans))

        if not no_sampling and is_down_sample:
            sample_block.append(self.down_sample)
        elif not no_sampling:  # Up-Sample
            sample_block.append(self.upsample(in_channels=outplanes, out_channels=outplanes, kernel_size=2, stride=2, dimension=D))

        return nn.ModuleList(sample_block)

    def _make_head(self, D, A, in_planes):
        def get_trans_planes(index):
            map_id = self.trans_map[index-self.num_down-1] - 1
            p = in_planes if map_id == -1 else cated_planes[map_id]
            return p

        def get_trans_blk(index):
            return self.num_trans_blks[index-self.num_down-1]

        def get_cur_planes(index):
            return self.network_planes[index]

        def get_blk_num(index):
            return self.num_res_blks[index]

        cated_planes, fish = [in_planes] * self.depth, []
        for i in range(self.depth):
            # even num for down-sample, odd for up-sample
            is_down, has_trans, no_sampling = i not in range(self.num_down, self.num_down+self.num_up+1),\
                i > self.num_down, i == self.num_down
            cur_planes, trans_planes, cur_blocks, num_trans =\
                get_cur_planes(i), get_trans_planes(i), get_blk_num(i), get_trans_blk(i)

            stg_args = [is_down, cated_planes[i - 1], cur_planes, cur_blocks]

            if is_down or no_sampling:
                k, dilation = 1, 1
            else:
                k, dilation = cated_planes[i - 1] // cur_planes, 2 ** (i-self.num_down-1)

            sample_block = self._make_stage(D, A, *stg_args, has_trans=has_trans, trans_planes=trans_planes,
                                            has_score=(i==self.num_down), num_trans=num_trans, k=k, dilation=dilation,
                                            no_sampling=no_sampling)
            if i == self.depth - 1:
                sample_block.extend(self._make_score(D, A, cur_planes + trans_planes, out_ch=self.num_cls, has_pool=True))
            elif i == self.num_down:
                sample_block.append(nn.Sequential(self._make_se_block(D, A, cur_planes*2, cur_planes)))

            if i == self.num_down-1:
                cated_planes[i] = cur_planes * 2
            elif has_trans:
                cated_planes[i] = cur_planes + trans_planes
            else:
                cated_planes[i] = cur_planes
            fish.append(sample_block)
        return nn.ModuleList(fish[7:])
    
    def _fish_forward(self, all_feat):
        def _concat(a, b):
            return torch.cat([a, b], dim=1)

        def stage_factory(*blks):
            def stage_forward(*inputs):
                if stg_id < self.num_down:  # tail
                    tail_blk = nn.Sequential(*blks[:2])
                    return tail_blk(*inputs)
                elif stg_id == self.num_down:
                    score_blks = nn.Sequential(*blks[:2])
                    score_feat = score_blks(inputs[0])
                    att_feat = blks[3](score_feat)
                    return blks[2](score_feat) * att_feat + att_feat
                else:  # refine
                    feat_trunk = blks[2](blks[0](inputs[0]))
                    feat_branch = blks[1](inputs[1])
                return _concat(feat_trunk, feat_branch)
            return stage_forward

        stg_id = 0
        # tail:
        while stg_id < self.depth:
            stg_blk_body = stage_factory(*self.head[stg_id])
            if stg_id <= self.num_down:
                in_feat = [all_feat[stg_id]]
            else:
                trans_id = self.trans_map[stg_id-self.num_down-1]
                in_feat = [all_feat[stg_id], all_feat[trans_id]]

            all_feat[stg_id + 1] = stg_blk_body(*in_feat)
            stg_id += 1
            # loop exit
            if stg_id == self.depth:
                score_feat = self.head[self.depth-1][-2](all_feat[-1])
                score = self.head[self.depth-1][-1](score_feat)
                return score

    def forward(self, x):
        all_feat = [None] * (self.depth + 1)
        all_feat[0] = x
        return self._fish_forward(all_feat)
    
    
class FishNet(ME.MinkowskiNetwork):
    def __init__(self, block, D, A, **kwargs):
        super(FishNet, self).__init__(D)

        inplanes = kwargs['network_planes'][0]
        # resolution: 224x224
        self.conv1 = self._conv_bn_relu(D, A, 3, inplanes // 2, stride=2)
        self.conv2 = self._conv_bn_relu(D, A, inplanes // 2, inplanes // 2)
        self.conv3 = self._conv_bn_relu(D, A, inplanes // 2, inplanes)
        self.pool1 = ME.MinkowskiMaxPooling(3, stride=2, dimension=D)
        # construct fish, resolution 56x56
        #self.fish = Fish(block, D, A, **kwargs)
        self.body = FishBody(block, D, A, **kwargs)
        self.head = FishHead(block, D, A, **kwargs)
        self.union = ME.MinkowskiUnion()
        self._init_weights()

    def _conv_bn_relu(self, D, A, in_ch, out_ch, stride=1):
        return nn.Sequential(ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, stride=stride,
                             bias=False, dimension=D),
                             ME.MinkowskiBatchNorm(out_ch),
                             minkowski_wrapper(D, A))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                n = m.kernel.data.shape[0] * m.kernel.data.shape[1] * m.out_channels
                m.kernel.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, ME.MinkowskiBatchNorm):
                m.bn.weight.data.fill_(1)
                m.bn.bias.data.zero_()

    def forward(self, x, device="cuda"):
        xview = ME.SparseTensor(x[0], x[1], device=device)
        yview = ME.SparseTensor(x[2], x[3], device=device, coordinate_manager=xview.coordinate_manager)
        print(xview.size())
        print(yview.size())
        
        xview = self.conv1(xview)
        xview = self.conv2(xview)
        xview = self.conv3(xview)
        xview = self.pool1(xview)
        
        yview = self.conv1(yview)
        yview = self.conv2(yview)
        yview = self.conv3(yview)
        yview = self.pool1(yview)
        
        score_body_x = self.body(xview)
        score_body_y = self.body(yview)
        
        x = self.union(score_body_x, score_body_y)
        
        score = self.head(x)
        # 1*1 output
        out = score.view(x.size(0), -1)

        return out


def fish(**kwargs):
    return FishNet(Bottleneck, **kwargs)