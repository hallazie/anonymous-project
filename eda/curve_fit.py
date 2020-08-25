# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: common.py
# @time: 2020/7/11 15:09
# @desc:

from config import logger, DATA_PATH_ROOT
from utils.common import normalize_vector, polynomial_interpolation, linear_interpolation
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import traceback
import os


class EDA:
    def __init__(self):
        self._init_data()

    def _init_data(self):
        self._fetch_meta_data()
        self._fetch_patient_files()

    def _fetch_meta_data(self):
        df = pd.read_csv('data/train.csv')
        header_idx = {k: i for i, k in enumerate(df.columns)}
        self.data = defaultdict(list)
        for row in df.values:
            uid = row[header_idx['Patient']]
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            age = row[header_idx['Age']]
            self.data[uid].append((week, fvc, percent, age))
        for uid in self.data:
            self.data[uid] = sorted(self.data[uid], key=lambda x: x[0])

    def _fetch_patient_files(self):
        self.train_user_dirs, self.test_user_dirs = [], []
        for d in os.listdir(os.path.join(DATA_PATH_ROOT, 'train')):
            path = os.path.join(DATA_PATH_ROOT, 'train', d)
            if not os.path.isdir(path):
                continue
            self.train_user_dirs.append(path)
        for d in os.listdir(os.path.join(DATA_PATH_ROOT, 'test')):
            path = os.path.join(DATA_PATH_ROOT, 'test', d)
            if not os.path.isdir(path):
                continue
            self.test_user_dirs.append(path)

    def _fit_all_fvc_curve(self):
        for uid in self.data:
            fvc_list = [x[1] for x in self.data[uid]]
            week_list = [x[0] for x in self.data[uid]]
            fvc_list_interpolated, div = polynomial_interpolation(week_list, fvc_list, power=3)
            week_list_ = [i for i in range(min(week_list), max(week_list) + 1)]
            plt.plot(week_list, fvc_list)
            plt.plot(week_list_, fvc_list_interpolated)
            plt.savefig('output/fvc-curve/%s-plot-pow3.png' % uid)
            print(uid, 'polynomial interpolation finished')
            plt.clf()

    def run(self):
        self._fit_all_fvc_curve()


if __name__ == '__main__':
    eda = EDA()
    eda.run()
