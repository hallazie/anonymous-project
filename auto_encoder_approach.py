# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: xgb_approach.py
# @time: 2020/12/21 21:22
# @desc:

from config import *
from score import scoring
from dataloader import DataLoader
from blocks import AutoEncoder

import torch
import torch.nn as nn
import numpy as np
import os
import warnings


warnings.filterwarnings("ignore")


loader = DataLoader()

auto_encoder = AutoEncoder(inp_size=130, out_size=5).cuda()
# auto_encoder = torch.load('checkpoints/autoencoder-base.pkl').cuda()
auto_encoder.train()

optimizer = torch.optim.Adam(params=auto_encoder.parameters(), lr=1e-3)
criterion1 = torch.nn.MSELoss()
criterion2 = torch.nn.BCELoss()

epoches = 1000
batchsize = 4096
datasize = loader.X_train.shape[0]

for e in range(epoches):
    for b in range(datasize//batchsize):
        auto_encoder.zero_grad()
        x, y = loader.X_train[b*batchsize:(b+1)*batchsize], loader.y_train[b*batchsize:(b+1)*batchsize]
        x = torch.tensor(x.values).float().cuda()
        # y = torch.unsqueeze(torch.tensor(y.values), 1).float().cuda()
        y = torch.tensor(y).float().cuda()
        p, d = auto_encoder(x)
        loss1 = criterion1(d, x)
        loss2 = criterion2(p, y)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        if b % 100 == 0:
            print(f'epoch={e}, batch={b}, mse={loss1.data}, bce={loss2.data} loss={loss.data}')
    if (e+1) % 20 == 0 and e > 0:
        torch.save(auto_encoder, f'checkpoints/autoencoder-{e+1}.pkl')


if __name__ == '__main__':
    pass

