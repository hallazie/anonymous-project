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
        train = dt.fread(TOTAL_PATH)
        train = train.to_pandas()
        train = train.query('date > 85').reset_index(drop=True)
        train = train.query('weight != 0').reset_index(drop=True)
        train.fillna(train.mean(), inplace=True)

        train['action'] = (train['resp'].values > 0).astype('int')
        resp_cols = ['resp', 'resp_1', 'resp_2', 'resp_3', 'resp_4']
        X_train = train.loc[:, train.columns.str.contains('feature')].values
        # X_train = train.loc[:, features].values
        # y_train = train.loc[:, resp_cols].values
        y_train = np.stack([(train[c] > 0).astype('int') for c in resp_cols]).T
        w_train = train.loc[:, 'weight']
        r_train = train.loc[:, 'resp']
        d_train = train.loc[:, 'date']

        y_train = y_train * np.tile(np.clip(w_train, a_min=0.25, a_max=8), (5, 1)).T

        self.X_train, self.X_test, self.y_train, self.y_test, self.w_train, self.w_test, self.r_train, self.r_test, self.d_train, self.d_test = train_test_split(X_train, y_train, w_train, r_train, d_train, random_state=RANDOM_SEED, test_size=0.05)

        print(f'train: {[e[:20] for e in self.X_train[:2]]}')
        print(f'test:  {[e[:20] for e in self.X_test[:2]]}')
        print(f'train and test set init finished with size: {self.X_train.shape} {self.y_train.shape}, {self.X_test.shape} {self.y_test.shape}')
        print(f'train date: {len(set(self.d_train))}, test date: {len(set(self.d_test))}, datatype={type(self.X_train)}')


if __name__ == '__main__':
    loader = DataLoader()


