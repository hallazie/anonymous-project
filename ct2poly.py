# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: ct2poly.py
# @time: 2020/8/1 17:19
# @desc:

from model.ct2polyfit import CT2PolyModel
from utils.dataloader import PolynomialFitRegressionDataset
from config import logger

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import torch
import datetime


class CT2PolynomialFit:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        logger.info('using device: %s' % self.device)
        self.bins = 4
        self.lr = 1e-3
        self.wd = 1e-4
        self.bs = 8
        self.ep = 1000
        self.checkpoint_path = 'checkpoint/ct2poly-%s.pkl' % datetime.date.today()
        self._init_data()
        self._init_model()

    def _init_data(self):
        self.train_db = PolynomialFitRegressionDataset(bins=self.bins)
        self.loader = DataLoader(self.train_db, batch_size=self.bs, shuffle=True)

    def _init_model(self):
        self.model = CT2PolyModel(self.bins)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = nn.MSELoss()
        torch.set_grad_enabled(True)
        self.model.train()

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
        logger.info('checkpoint [%s] saved') % self.checkpoint_path

    def train(self):
        for e in range(self.ep):
            for i, (x, y) in enumerate(self.loader):
                x = torch.from_numpy(x).to(self.device)
                y = torch.from_numpy(y).to(self.device)
                p = self.model(x)
                loss = self.criterion(p, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info('epoch %s batch %s, with MSE loss: %s' % (e, i, loss.data.item()))
        self._save_checkpoint()


if __name__ == '__main__':
    ct2poly = CT2PolynomialFit()
    ct2poly.train()
