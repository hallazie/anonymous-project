# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: eda.py
# @time: 2020/7/11 15:09
# @desc:

from config import logger, DATA_PATH_ROOT

import pandas as pd
import numpy as np
import pydicom
import seaborn as sns
import random
import os


class EDA:
    def __init__(self):
        self._init_data()

    def _init_data(self):
        self._fetch_patient_files()

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

    def _visualize_random_single(self):
        d_idx = random.randint(0, len(self.train_user_dirs))
        f_list = os.listdir(self.train_user_dirs[d_idx])
        f_idx = random.randint(0, len(f_list))
        path = os.path.join(self.train_user_dirs[d_idx], f_list[f_idx])



if __name__ == '__main__':
    eda = EDA()