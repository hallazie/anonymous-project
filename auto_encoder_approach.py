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


def train():
    loader = DataLoader()
    epoches = 500
    batchsize = 4096
    datasize = loader.X_train.shape[0]
    auto_encoder = AutoEncoder(inp_size=130, out_size=5, batchsize=batchsize).cuda()
    # auto_encoder = torch.load('checkpoints/autoencoder-base.pkl').cuda()
    auto_encoder.train()
    optimizer = torch.optim.Adam(params=auto_encoder.parameters(), lr=1e-3)
    criterion1 = torch.nn.MSELoss()
    # criterion2 = torch.nn.BCELoss()
    for ep in range(epoches):
        for b in range(datasize//batchsize):
            optimizer.zero_grad()
            x, _ = loader.X_train[b*batchsize:(b+1)*batchsize], loader.y_train[b*batchsize:(b+1)*batchsize]
            x = torch.tensor(x).float().cuda()
            _, d = auto_encoder(x)
            loss1 = criterion1(d, x)
            # loss2 = criterion2(p, y)
            # loss = loss1 + loss2
            loss1.backward()
            optimizer.step()
            if b % 100 == 0:
                # print(f'epoch={e}, batch={b}, mse={loss1.data}, bce={loss2.data} loss={loss.data}')
                print(f'epoch={ep}, batch={b}, mse={loss1.data}')
        if (ep+1) % 100 == 0 and ep > 0:
            torch.save(auto_encoder, f'checkpoints/autoencoder-{ep+1}.pkl')


def test():
    import pandas as pd
    model = torch.load('checkpoints/autoencoder-100.pkl')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    # df = pd.read_csv('data/example_train.csv')
    df = pd.read_csv('data/example_test.csv')
    # df = df.query('date > 85').reset_index(drop=True)
    df = df.fillna(df.mean())
    df = df.query('weight != 0').reset_index(drop=True)
    feature = df.loc[:, df.columns.str.contains('feature')].values
    feature = torch.from_numpy(feature).float().cuda()
    encoded, denoise = model(feature)
    for i in range(5):
        print(feature[i])
        print(denoise[i])
        print('---------------------')
    criterion = nn.MSELoss()
    loss = criterion(feature, denoise)
    print(loss.data)


if __name__ == '__main__':
    train()

