# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: blocks.py
# @time: 2020/8/1 16:49
# @desc: blocks

import torch
import torch.nn as nn


def Conv2d(c_in, c_out, kernel_size, pad, act='relu'):
    seq = [
        nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
        nn.BatchNorm2d(c_out)
    ]
    if act == 'relu':
        seq.append(nn.ReLU6(inplace=True))
    elif act == 'tanh':
        seq.append(nn.Tanh())
    return nn.Sequential(*seq)


def Pool(type_, kernel_size=2, stride=2):
    if type_ == 'max':
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
    elif type_ == 'avg':
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride)


def Dense(c_in, c_out, bias=True):
    return nn.Sequential(
        nn.Linear(in_features=c_in, out_features=c_out, bias=bias),
    )
