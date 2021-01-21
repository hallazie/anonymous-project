# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: xgb_approach.py
# @time: 2020/12/21 21:22
# @desc:

from sklearn.model_selection import train_test_split
from config import *
from score import scoring
from blocks import *

import torch
import torch.nn as nn
import numpy as np
import datatable as dt
import os

train = dt.fread(TRAIN_PATH)
train = train.to_pandas()
train = train[train['weight'] != 0]
train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']
w_train = train.loc[:, 'weight']
r_train = train.loc[:, 'resp']
d_train = train.loc[:, 'date']

X_train, X_test, y_train, y_test, w_train, w_test, r_train, r_test, d_train, d_test = train_test_split(X_train, y_train, w_train, r_train, d_train, random_state=666, test_size=0.08)
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f'train and test set init finished with size: {X_train.shape} {y_train.shape}, {X_test.shape} {y_test.shape}')
print(f'train date: {len(set(d_train))}, test date: {len(set(d_test))}, datatype={type(X_train)}')


auto_encoder = AutoEncoder(inp_size=130, out_size=1)

optimizer = torch.optim.Adam(params=auto_encoder.parameters())
criterion = torch.nn.MSELoss()




if __name__ == '__main__':
    pass

