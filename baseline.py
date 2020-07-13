# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: baseline.py
# @time: 2020/7/11 22:30
# @desc: baselines for the task

from sklearn.linear_model import Ridge
from collections import defaultdict
from utils.evaluate import evaluate_on_testset
from utils.dataloader import DataLoader

import lightgbm as lgb
import torch
import pandas as pd


class Baseline:
    def __init__(self):
        self.loader = DataLoader()
        self._init_data()

    def _init_data(self):
        self.trainset = defaultdict(list)
        self.testset = defaultdict(list)
        self._load_basic('data/train.csv', self.trainset)
        self._load_basic('data/test.csv', self.testset)
        self.x_train, self.y_train, self.x_val, self.y_val = self.loader.get_dataset(fold=0.9)

    @staticmethod
    def _load_basic(path, dataset):
        df = pd.read_csv(path)
        header_idx = {k: i for i, k in enumerate(df.columns)}
        for row in df.values:
            uid = row[header_idx['Patient']]
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            dataset[uid].append((week, fvc, percent))

    def _baseline_average(self):
        self._init_data()
        output = []
        for uid in self.testset:
            inp = self.testset[uid][0]
            for i in range(-12, 134):
                uid_week = '%s_%s' % (uid, i)
                output.append((uid_week, inp[1], 833))
        result = evaluate_on_testset(output)
        print('average result <laplace log-likelihood>: %s' % result)

    @staticmethod
    def _baseline_quantile_regression():
        output = []
        for row in pd.read_csv('output/submission-quantile-regression.csv').values:
            output.append((row[0], row[1], row[2]))
        result = evaluate_on_testset(output)
        print('average result <laplace log-likelihood>: %s' % result)

    def _baseline_xgboost(self):
        pass

    def _baseline_light_gbm(self):
        model = lgb.LGBMRegressor()
        model.fit(self.x_train, self.y_train)

    def _baseline_basic_cnn(self):
        pass

    def run(self):
        self._baseline_light_gbm()


if __name__ == '__main__':
    baseline = Baseline()
    baseline.run()
