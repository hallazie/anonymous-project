# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: ct2poly.py
# @time: 2020/8/1 17:19
# @desc:

from model.ct2polyfit_model import CT2PolyModel
from utils.dataloader import PolynomialFitRegressionDataset
from config import logger

from torch import optim
from torch import nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import datetime


class CT2PolynomialFit:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        logger.info('using device: %s' % self.device)
        self.bins = 4
        self.lr = 1e-3
        self.wd = 1e-4
        self.bs = 8
        self.ep = 5000
        self.checkpoint_path = 'checkpoints/ct2poly-%s.pkl' % datetime.date.today()
        self.meta_path = 'checkpoints/polynomial-pow3.json'

    def _init_data(self, train=True):
        self.train_db = PolynomialFitRegressionDataset(bins=self.bins, train=train)
        self.loader = DataLoader(self.train_db, batch_size=self.bs, shuffle=True)

    def _init_model(self):
        self.model = CT2PolyModel(self.bins)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = nn.MSELoss()
        torch.set_grad_enabled(True)
        self.model.train()

    def _load_model(self):
        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            self.avg = np.array(meta['avg'])
            self.std = np.array(meta['std'])
        self.model = CT2PolyModel(self.bins)
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(state_dict=checkpoint, strict=True)

    def _save_checkpoint(self):
        torch.save(self.model.state_dict(), self.checkpoint_path)
        logger.info('checkpoint [%s] saved' % self.checkpoint_path)

    def train(self):
        self._init_data(True)
        self._init_model()
        for e in range(self.ep):
            for i, (x, y) in enumerate(self.loader):
                x = x.to(self.device)
                y = y.to(self.device).float()
                p = self.model(x)[:, :, 0, 0].float()
                loss = self.criterion(p, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                logger.info('epoch %s batch %s, with MSE loss: %s' % (e, i, loss.data.item()))
                if i == 0:
                    logger.info('epoch %s example label: %s, predict: %s' % (e, str(y[0].detach()), str(p[0].detach())))
        self._save_checkpoint()

    @staticmethod
    def translate_polynomial(coefficient):
        poly = np.poly1d(coefficient)
        x = [i for i in range(-20, 120)]
        y = poly(x)
        return x, y

    def inference(self, x):
        p = self.model(x)[0, :, 0, 0].detach().cpu().detach().numpy()
        p = (p * self.std) + self.avg
        return p

    def test(self):
        self.bs = 1
        self._init_data(False)
        self._load_model()
        for i, (x, y) in enumerate(self.loader):
            if i == 4:
                break
            y = y.cpu().numpy()[0]
            x = x.to(self.device)
            p = self.inference(x)
            x0, y0 = self.translate_polynomial(y)
            x1, y1 = self.translate_polynomial(p)
            plt.subplot('22%s' % (i+1))
            print('y: %s, p: %s' % (str(y), str(p)))
            plt.plot(x0, y0)
            plt.plot(x1, y1)
        plt.show()


if __name__ == '__main__':
    ct2poly = CT2PolynomialFit()
    ct2poly.test()
