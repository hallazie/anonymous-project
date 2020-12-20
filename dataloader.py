# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: dataloader.py
# @time: 2020/12/19 18:21
# @desc:

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
        df = dt.fread(TRAIN_PATH)
        df = df.to_pandas().fillna(0)
        self.data = df

    @staticmethod
    def _normalize_x(df):
        for col in df:
            val = df[col]
            val = (val-np.min(val)) / (np.max(val)-np.min(val))
            df[col] = val
        return df

    def get_train_and_test(self, fold=10, shuffle=False, train=True):
        date_list = sorted(set(self.data.date))
        if shuffle:
            random.shuffle(date_list)
        train_size = int(len(date_list) * ((fold - 1) / float(fold)))
        date_set = date_list[:train_size] if train else date_list[train_size:]
        col_names = self.data.columns
        total_x = self.data[[col for col in col_names if col.startswith('feature')]]
        # total_x = np.array(self._normalize_x(total_x))
        total_x = np.array(total_x)
        total_y = np.expand_dims(np.array((self.data.resp > 0).astype('int')), 1)
        total_r = np.array(self.data.resp)
        total_d = np.array(self.data.date)
        total_w = np.array(self.data.weight)
        ret_x = total_x[np.isin(total_d, date_set)]
        ret_y = total_y[np.isin(total_d, date_set)]
        ret_d = total_d[np.isin(total_d, date_set)]
        ret_r = total_r[np.isin(total_d, date_set)]
        ret_w = total_w[np.isin(total_d, date_set)]
        return ret_x, ret_y, ret_d, ret_r, ret_w

    def basic_eval(self):
        weight = list(self.data.weight)
        resp = list(self.data.resp)
        date = list(self.data.date)
        action = [1 if resp[i] > 0 else 0 for i in range(len(weight))]
        score, t = scoring(date_list=date, weight_list=weight, resp_list=resp, action_list=action)
        print(f'final score (size={len(resp)}) for random action: {score}, with t={t}')


if __name__ == '__main__':
    loader = DataLoader()
    # loader.basic_eval()


