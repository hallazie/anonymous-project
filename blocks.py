# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: blocks.py
# @time: 2021/1/21 23:27
# @desc:

import torch
import torch.nn as nn
import numpy as np


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.05):
        super().__init__()
        self.noise = torch.from_numpy(np.zeros(shape, shape)).cuda()
        self.std = std

    def forward(self, x):
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)

        print(x.size(), self.noise.size())
        return x + self.noise


class AutoEncoder(nn.Module):
    def __init__(self, inp_size, out_size, noise=0.05):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(inp_size),
            # DynamicGNoise((inp_size, )),
            nn.Linear(in_features=inp_size, out_features=64),
            nn.ReLU6(inplace=False),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=inp_size)
        )
        self.more = nn.Sequential(
            nn.Linear(in_features=inp_size, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        out = self.more(dec)
        return out, enc





