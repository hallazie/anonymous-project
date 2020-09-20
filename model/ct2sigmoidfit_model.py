# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: ct2sigmoidfit_model.py
# @time: 2020/9/19 21:16
# @desc:

from model.blocks import *

import torch
import torch.nn as nn


class CT2SigmoidModel(nn.Module):
    def __init__(self, c_in):
        super(CT2SigmoidModel, self).__init__()
        self.backbone = nn.Sequential(
            Conv2d(c_in, 16, 3, 1, act='leaky'),
            Pool('max'),            # 256
            Conv2d(16, 32, 3, 1, act='leaky'),
            Conv2d(16, 32, 3, 1, act='leaky'),
            Pool('max'),            # 128
            Conv2d(32, 64, 3, 1, act='leaky'),
            Conv2d(32, 64, 3, 1, act='leaky'),
            Pool('max'),            # 64
            Conv2d(64, 128, 3, 1, act='leaky'),
            Conv2d(64, 128, 3, 1, act='leaky'),
            Pool('max'),            # 32
            Conv2d(128, 256, 3, 1, act='leaky'),
            Conv2d(128, 256, 3, 1, act='leaky'),
            Pool('max'),            # 16
            Conv2d(256, 512, 3, 1, act='leaky'),
            Conv2d(256, 512, 3, 1, act='leaky'),
            Pool('max'),            # 8
            Conv2d(512, 512, 3, 1, act='leaky'),
            Conv2d(512, 512, 3, 1, act='leaky'),
            Pool('max'),            # 4
            Conv2d(512, 4, 3, 1, act='leaky'),
            Pool('max', kernel_size=32 * 3, stride=4),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


