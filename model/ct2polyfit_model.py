# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: ct2polyfit_model.py
# @time: 2020/8/1 16:44
# @desc: 均匀采样后的ct-scan序列预测fvc序列的polynomial-fit系数

from model.blocks import *

import torch
import torch.nn as nn


class CT2PolyModel(nn.Module):
    def __init__(self, c_in):
        super(CT2PolyModel, self).__init__()
        self.backbone = nn.Sequential(
            Conv2d(c_in, 16, 3, 1),
            Pool('max'),            # 256
            Conv2d(16, 32, 3, 1),
            Pool('max'),            # 128
            Conv2d(32, 64, 3, 1),
            Pool('max'),            # 64
            Conv2d(64, 128, 3, 1),
            Pool('max'),            # 32
            Conv2d(128, 256, 3, 1),
            Pool('max'),            # 16
            Conv2d(256, 512, 3, 1, act='none'),
            Pool('max'),            # 8
            Conv2d(512, 512, 3, 1, act='none'),
            Pool('max'),            # 4
            Conv2d(512, 4, 3, 1, act='none'),
            Pool('max', kernel_size=4, stride=4),
            # Dense(512, 512, True),
            # Dense(512, 4, True),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
