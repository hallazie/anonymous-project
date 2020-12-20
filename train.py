# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2020/12/20 14:05
# @desc:

from dataloader import DataLoader
from mlp_model import MLPModel
from config import LOGGER, LAYERS
from test import Tester

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np


class Train:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.epoch = 512
        self.batch_size = 4096
        self.lr = 1e-3
        self.loader = DataLoader()

    def train(self):
        torch.set_grad_enabled(True)
        logfile = open('log', 'w')
        logfile.close()
        logfile = open('log', 'a', encoding='utf-8')
        train_x, train_y, _, _, _ = self.loader.get_train_and_test(10, shuffle=False, train=True)
        model = MLPModel(130, LAYERS, 1).to(self.device)
        # loss = nn.BCEWithLogitsLoss()
        loss = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        for e in range(self.epoch):
            for b in range(len(train_y) // self.batch_size):
                optimizer.zero_grad()
                x = train_x[b*self.batch_size:(b+1)*self.batch_size]
                y = train_y[b*self.batch_size:(b+1)*self.batch_size]
                u = [i for i in range(len(y))]
                random.shuffle(u)
                u = np.array(u)
                x = x[u]
                y = y[u]
                x = torch.from_numpy(x).to(self.device).float()
                y = torch.from_numpy(y).to(self.device).float()
                p = model(x).float()
                l = loss(p, y)
                l.backward()
                optimizer.step()
                action = p[:, 0].cpu().detach().numpy()
                act_buy = sum([1 if p_ >= 0.5 else 0 for p_ in action])
                act_pas = len(action) - act_buy
                LOGGER.info(f'epoch={e}, step={b}, buy/pass={act_buy}/{act_pas}, loss={l.data.item()}')
                logfile.write(f'epoch={e}, step={b}, loss={l.data.item()}\n')
            torch.save(model.state_dict(), 'checkpoints/baseline.pkl')
        logfile.close()


if __name__ == '__main__':
    trainer = Train()
    trainer.train()



