# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: dataloader.py
# @time: 2020/7/14 0:23
# @desc:

from collections import defaultdict
from sklearn.utils import shuffle
from config import RANDOM_STATE
from utils.common import get_abs_path

import pandas as pd
import numpy as np
import os


class DataLoader:
    def __init__(self):
        self.train_path = get_abs_path(__file__, -2, 'data', 'train.csv')
        self.test_path = get_abs_path(__file__, -2, 'data', 'test.csv')
        self._init_raw()

    def _init_raw(self):
        test_uid = [row[0] for row in pd.read_csv(self.test_path).values]
        df = pd.read_csv(self.train_path)
        header_idx = {k: i for i, k in enumerate(df.columns)}
        data = defaultdict(list)
        for row in df.values:
            uid = row[header_idx['Patient']]
            if uid in test_uid:
                continue
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            age = row[header_idx['Age']]
            gender = row[header_idx['Sex']]
            smoker = row[header_idx['SmokingStatus']]
            data[uid].append((week, fvc, percent, age, gender, smoker))
        self.X, self.y = [], []
        for uid in data:
            basic = sorted(data[uid], key=lambda x: x[0])[0]
            basic_week = basic[0]
            basic_fvc = basic[1]
            basic_percent = basic[2]
            for x in data[uid]:
                week, fvc, percent, age, gender, smoker = x
                self.X.append((basic_week, basic_fvc, basic_percent, week, age, gender, smoker))
                self.y.append(fvc)

    def get_dataset(self, fold=0.9):
        X, y = shuffle(self.X, self.y, random_state=RANDOM_STATE)
        size = int(float(fold) / 1. * len(X))
        print(size)
        # return X[:size], y[:size], X[size:], y[size:]
        return np.array(X[:size]), np.array(y[:size]), np.array(X[size:]), np.array(y[size:])


if __name__ == '__main__':
    # loader = DataLoader()
    # x0, y0, x1, y1 = loader.get_dataset()
    # print(len(x0), len(y0), len(x1), len(y1))
    # print(x1)
    # print(y1)
    print(__file__)