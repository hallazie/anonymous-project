# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: test.py
# @time: 2020/12/20 17:18
# @desc:

from dataloader import DataLoader
from mlp_model import MLPModel
from config import LOGGER, LAYERS
from score import scoring

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Tester:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2048
        self.loader = DataLoader()
        self._init_data()

    def _load_model(self):
        self.model = MLPModel(130, LAYERS, 1).to(self.device)
        self.model.eval()
        checkpoint = torch.load('checkpoints/baseline.pkl')
        self.model.load_state_dict(state_dict=checkpoint, strict=True)

    def _init_data(self):
        self._load_model()
        self.test_x, self.test_y, self.test_d, self.test_r, self.test_w = self.loader.get_train_and_test(10, shuffle=False, train=False)

    def test(self, print_info=False):
        w_list, r_list, a_list, d_list, p_list = [], [], [], [], []
        for b in range(len(self.test_x) // self.batch_size):
            x = self.test_x[b * self.batch_size:(b + 1) * self.batch_size]
            y = self.test_y[b * self.batch_size:(b + 1) * self.batch_size]
            d = self.test_d[b * self.batch_size:(b + 1) * self.batch_size]
            r = self.test_r[b * self.batch_size:(b + 1) * self.batch_size]
            w = self.test_w[b * self.batch_size:(b + 1) * self.batch_size]

            x = torch.from_numpy(x).to(self.device).float()
            y = torch.from_numpy(y).to(self.device).float()
            p = self.model(x).float()[:, 0].cpu().detach().numpy()
            a = y.cpu().detach().numpy()
            w_list.extend(list(w))
            r_list.extend(list(r))
            a_list.extend(list(a))
            p_list.extend(list(p))
            d_list.extend(list(d))
        score_pred = scoring(date_list=d_list, weight_list=w_list, resp_list=r_list, action_list=p_list)
        score_perf = scoring(date_list=d_list, weight_list=w_list, resp_list=r_list, action_list=a_list)
        p_list = [1 if p_ >= 0 else 0 for p_ in p_list]
        a_list = [1 if a_ >= 0 else 0 for a_ in a_list]
        acc = sum([1 if p_ == y_ else 0 for p_, y_ in zip(p_list, a_list)]) / float(len(p_list))
        if print_info:
            print(f'predict: {p_list[:1024]}')
            print(f'labels:  {[int(a_) for a_ in a_list[:1024]]}')
            print(f'final score for test set is: {score_pred}, with accuracy: {acc}, while perfect score={score_perf}')
            conf_pred = sum(p_list) / float(len(p_list))
            conf_perf = sum(a_list) / float(len(a_list))
            print(f'precit conficence={conf_pred}, while perfect confidence={conf_perf}')
        return score_pred


if __name__ == '__main__':
    tester = Tester()
    tester.test(print_info=True)



