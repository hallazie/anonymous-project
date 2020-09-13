# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: baseline.py
# @time: 2020/7/11 22:30
# @desc: baselines for the task

from collections import defaultdict
from utils.evaluate import evaluate_on_testset, evaluate_on_array
from dataloader.fvc_simple_loader import FVCPredictDataLoader
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from config import logger

import torch
import time
import pandas as pd
import numpy as np


class Baseline:
    def __init__(self):
        self.loader = FVCPredictDataLoader()
        self._init_data()

    def _init_data(self):
        self.trainset = defaultdict(list)
        self.testset = defaultdict(list)
        self._load_basic('data/train.csv', self.trainset)
        self._load_basic('data/test.csv', self.testset)
        # self.x_train, self.y_train, self.x_val, self.y_val = self.loader.get_dataset(fold=0.9)
        self.x_train, self.y_train, self.x_val, self.y_val = self.loader.get_dataset_with_ct(bins=50, fold=0.9)
        # self.x_train, self.y_train, self.uid_train, self.x_val, self.y_val, self.uid_val = self.loader.get_dataset_with_uid(fold=0.9)
        logger.info('training size: %s, %s, validation size: %s, %s' % (str(self.x_train.shape), str(self.y_train.shape), str(self.x_val.shape), str(self.y_val.shape)))

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
        logger.info('average result <laplace log-likelihood>: %s' % result)

    @staticmethod
    def _baseline_combined_model():
        pass

    @staticmethod
    def _baseline_quantile_regression():
        """
        eval on notebook result
        :return:
        """
        output = []
        for row in pd.read_csv('output/submission-quantile-regression.csv').values:
            output.append((row[0], row[1], row[2]))
        result = evaluate_on_testset(output)
        logger.info('average result <laplace log-likelihood>: %s' % result)

    def _baseline_xgboost(self):
        model = XGBRegressor()
        start = time.time()
        model.fit(self.x_train, self.y_train, verbose=True)
        end = time.time()
        logger.info('training finished with %ss' % (end - start))
        predict = model.predict(self.x_val)
        confidence = [np.std([self.y_val[i], predict[i]]) for i in range(len(self.y_val))]
        avg_metric = evaluate_on_array(self.y_val, predict, confidence)
        logger.info('baseline with XGBoost laplace log-likelihood: %s' % avg_metric)

    def _baseline_light_gbm(self):
        model = LGBMRegressor(objective='regression', verbose=0)
        start = time.time()
        model.fit(self.x_train, self.y_train, verbose=True)
        end = time.time()
        logger.info('training finished with %ss' % (end - start))
        predict = model.predict(self.x_val)
        confidence = [np.std([self.y_val[i], predict[i]]) for i in range(len(self.y_val))]
        avg_metric = evaluate_on_array(self.y_val, predict, confidence)
        logger.info('baseline with LightGBM laplace log-likelihood: %s' % avg_metric)

    def _baseline_basic_cnn(self):
        pass

    def run(self):
        """
        with 32*32 ct feature: *
        with only meta feature: -6.385669612878189
        with 32*32 ct feature samples in even 10 bins: -6.256543006505297
        with 32*32 ct feature samples in even 20 bins: -6.261188326751937
        with 32*32 ct feature samples in even 20 bins: -6.320450703147999
        :return:
        """
        # self._baseline_xgboost()
        self._baseline_light_gbm()


if __name__ == '__main__':
    baseline = Baseline()
    baseline.run()
