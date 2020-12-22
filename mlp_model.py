# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: mlp_model.py
# @time: 2020/12/20 13:57
# @desc:

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_size, neural_size, label_size):
        super(MLPModel, self).__init__()
        neural_size_fill = [input_size] + neural_size
        neural_list = [
            (nn.Linear(in_features=neural_size_fill[i], out_features=neural_size_fill[i + 1], bias=True),
             nn.BatchNorm1d(num_features=neural_size_fill[i + 1]),
             nn.Dropout(0.125),
             nn.Tanh())
            for i in range(len(neural_size_fill) - 1)]
        neural_list = [x for y in neural_list for x in y]
        self.bone = nn.Sequential(*neural_list)
        self.out = nn.Sequential(nn.Linear(in_features=neural_size[-1], out_features=1), nn.Sigmoid())
        # self.out = nn.Sequential(nn.Linear(in_features=neural_size[-1], out_features=1))

    def forward(self, x):
        y = self.bone(x)
        y = self.out(y)
        return y

