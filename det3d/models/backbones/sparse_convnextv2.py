# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .utils import (
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from MinkowskiEngine.MinkowskiConvolution import MinkowskiConvolution
from MinkowskiEngine.MinkowskiDepthwiseConvolution import MinkowskiDepthwiseConvolution
from MinkowskiEngine.MinkowskiNonlinearity import MinkowskiGELU

from MinkowskiEngine.MinkowskiOps import MinkowskiLinear

class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., D=3):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4 * dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 D=3):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        # stem = nn.Sequential(
        #     nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        #     LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # )
        # self.downsample_layers.append(stem)
        for i in range(4):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i + 1], kernel_size=2, stride=2, bias=True, dimension=D)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(5):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    def forward(self, x):
        # torch.cuda.empty_cache()
        # multi_scale_x = []
        x0 = x
        x0 = self.stages[0](x0)

        x1 = self.downsample_layers[0](x0) #720×720->360×360
        x1 = self.stages[1](x1)

        x2 = self.downsample_layers[1](x1) #360×360->180×180
        x2 = self.stages[2](x2)
        # multi_scale_x.append(x2.dense()[0])

        x3 = self.downsample_layers[2](x2) #180×180->90×90
        x3 = self.stages[3](x3)
        # multi_scale_x.append(x3.dense()[0])

        x4 = self.downsample_layers[3](x3) #90×90->45×45
        x4 = self.stages[4](x4)
        # multi_scale_x.append(x4.dense()[0])

        return [x2.dense()[0], x3.dense()[0], x4.dense()[0]]
