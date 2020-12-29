# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2020/12/20 14:05
# @desc:

from sklearn.model_selection import train_test_split
from dataloader import DataLoader
from mlp_model import MLPModel
from config import *
from test import Tester
from score import scoring

import datatable as dt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class Train:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.epoch = 4096
        self.batch_size = 4096
        self.lr = 1e-3
        # self.loader = DataLoader()

    def train(self):
        torch.set_grad_enabled(True)
        logfile = open('log', 'w')
        logfile.close()
        logfile = open('log', 'a', encoding='utf-8')
        # train_x, train_y, _, _, _ = self.loader.get_train_and_test(10, shuffle=False, train=True)

        train = dt.fread(TRAIN_PATH)
        train = train.to_pandas()
        train = train[train['weight'] != 0]
        train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

        x_train = train.loc[:, train.columns.str.contains('feature')]
        y_train = train.loc[:, 'action']
        w_train = train.loc[:, 'weight']
        r_train = train.loc[:, 'resp']
        d_train = train.loc[:, 'date']

        train_x, test_x, train_y, test_y, train_w, test_w, train_r, test_r, train_d, test_d = train_test_split(x_train, y_train, w_train, r_train, d_train, random_state=666, test_size=0.1)
        train_x = train_x.fillna(0)
        test_x = test_x.fillna(0)

        print(f'training size: {train_x.shape}, testing size: {test_x.shape}')

        model = MLPModel(130, LAYERS).to(self.device)
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.BCELoss()
        # loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for e in range(self.epoch):
            for b in range(len(train_y) // self.batch_size):
                optimizer.zero_grad()
                x = train_x[b*self.batch_size:(b+1)*self.batch_size]
                y = train_y[b*self.batch_size:(b+1)*self.batch_size]
                x = torch.from_numpy(np.array(x)).to(self.device).float()
                y = torch.from_numpy(np.array(y)).to(self.device).float()
                p = model(x).float()
                l = loss(p, y)

                action = p[:, 0].cpu().detach().numpy()
                act_buy = sum([1 if p_ > 0.5 else 0 for p_ in action])
                act_pas = len(action) - act_buy
                act_rat = float(act_buy) / len(action)
                z = y.cpu().detach().numpy()
                acc = sum([1 if (1 if action[i] > 0.5 else 0) == z[i] else 0 for i in range(len(action))]) / float(len(action))

                l.backward()
                optimizer.step()

                LOGGER.info(f'epoch={e}, step={b}, buy/pass={act_buy}/{act_pas}={act_rat}, loss={l.data.item()}, acc={acc}')
                logfile.write(f'epoch={e}, step={b}, loss={l.data.item()}, acc={acc}\n')
            torch.save(model.state_dict(), 'checkpoints/baseline.pkl')
        logfile.close()

        pred = model(torch.from_numpy(np.array(test_x)).to(self.device).float())[:, 0].cpu().detach().numpy()
        print(pred.shape)
        score = scoring(date_list=list(test_d), weight_list=list(test_w), resp_list=list(test_r), action_list=list(pred))
        print(score)

    def test(self):
        model = MLPModel(130, LAYERS).to(self.device)
        model.eval()
        checkpoint = torch.load('checkpoints/baseline.pkl')
        model.load_state_dict(state_dict=checkpoint, strict=True)

        train = dt.fread(TRAIN_PATH)
        train = train.to_pandas()
        train = train[train['weight'] != 0]
        train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

        feature = [False] * 7 + [True] * 130 + [False] * 2
        # x_train = train.loc[:, train.columns.str.contains('feature')]
        x_train = train.loc[:, feature]
        y_train = train.loc[:, 'action']
        w_train = train.loc[:, 'weight']
        r_train = train.loc[:, 'resp']
        d_train = train.loc[:, 'date']

        train_x, test_x, train_y, test_y, train_w, test_w, train_r, test_r, train_d, test_d = train_test_split(x_train, y_train, w_train, r_train, d_train, random_state=666, test_size=0.1)
        train_x = train_x.fillna(0)
        test_x = test_x.fillna(0)
        print(test_x.shape)
        pred = model(torch.from_numpy(np.array(test_x)).to(self.device).float())[:, 0].cpu().detach().numpy()
        print(pred.shape)
        score = scoring(date_list=list(test_d), weight_list=list(test_w), resp_list=list(test_r), action_list=list(pred))
        print(score)


if __name__ == '__main__':
    trainer = Train()
    trainer.train()



