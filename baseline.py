# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: baseline.py
# @time: 2020/7/11 22:30
# @desc: baselines for the task

import xgboost
import torch


class Baseline:
    def __init__(self):
        pass

    def _baseline_xgboost(self):
        pass

    def _baseline_basic_cnn(self):
        pass

    def run(self):
        self._baseline_xgboost()


if __name__ == '__main__':
    baseline = Baseline()
