# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: dataloader.py
# @time: 2020/12/19 18:21
# @desc:

from sklearn.model_selection import train_test_split
from config import *
from utils import *
from score import scoring
from sklearn import preprocessing

import datatable as dt
import pandas as pd
import numpy as np
import random

random.seed = RANDOM_SEED


class DataLoader:
    def __init__(self):
        self._init_data()

    def _init_data(self):
        train = dt.fread(TRAIN_PATH)
        train = train.to_pandas()
        # train = pd.read_csv(TRAIN_PATH)
        train = train.query('date > 85').reset_index(drop=True)
        train.fillna(train.mean(), inplace=True)
        # train = train[train['weight'] != 0]
        train = train.query('weight != 0').reset_index(drop=True)

        # train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')
        train['action'] = (train['resp'].values > 0).astype('int')
        # f_mean = np.mean(train[[c for c in train.columns if 'feature' in c][1:]].values, axis=0)

        resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
        X_train = train.loc[:, train.columns.str.contains('feature')].values
        y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
        # y_train = train.loc[:, train.columns.str.contains('resp')]
        # y_train = train.loc[:, 'resp'] * train.loc[:, 'weight']
        w_train = train.loc[:, 'weight']
        r_train = train.loc[:, 'resp']
        d_train = train.loc[:, 'date']

        self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test, self.r_train, self.r_test, self.d_train, self.d_test = train_test_split(
            X_train, y_train, w_train, r_train, d_train, random_state=RANDOM_SEED, test_size=0.08)

        # self.X_train, self.X_test = X_train, X_train
        # self.y_train, self.y_test = y_train, y_train
        # self.w_train, self.w_test = w_train, w_train
        # self.r_train, self.r_test = r_train, r_train
        # self.d_train, self.d_test = d_train, d_train

        print(train.shape, self.X_train.shape, self.X_test.shape)
        print(train.mean().shape, self.X_train.mean().shape, self.X_test.mean().shape)

        print(f'train and test set init finished with size: {self.X_train.shape} {self.y_train.shape}, {self.X_test.shape} {self.y_test.shape}')
        print(f'train date: {len(set(self.d_train))}, test date: {len(set(self.d_test))}, datatype={type(self.X_train)}')


if __name__ == '__main__':
    loader = DataLoader()
