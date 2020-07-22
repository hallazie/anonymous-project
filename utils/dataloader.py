# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: dataloader.py
# @time: 2020/7/14 0:23
# @desc:

from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from config import RANDOM_STATE, logger
from ct.embedding import embedding
from ct.loader import loader
from ct.transform import transform_feature
from utils.common import get_abs_path


class DataLoader:
    def __init__(self):
        self.train_path = get_abs_path(__file__, -2, 'data', 'train.csv')
        self.test_path = get_abs_path(__file__, -2, 'data', 'test.csv')
        self.smoker_map = {'Never smoked': 0, 'Ex-smoker': 1, 'Currently smokes': 2}
        self._init_raw()

    def _init_raw(self):
        test_uid = [row[0] for row in pd.read_csv(self.test_path).values]
        df = pd.read_csv(self.train_path)
        header_idx = {k: i for i, k in enumerate(df.columns)}
        data = defaultdict(list)
        for row in df.values:
            uid = row[header_idx['Patient']]
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            age = row[header_idx['Age']]
            gender = row[header_idx['Sex']]
            smoker = row[header_idx['SmokingStatus']]
            gender_ = 0 if gender == 'Male' else 1
            smoker_ = self.smoker_map[smoker]
            data[uid].append((week, fvc, percent, age, gender_, smoker_))
        self.X, self.y, self.uid = [], [], []
        for uid in data:
            basic = sorted(data[uid], key=lambda x: x[0])[0]
            basic_week = basic[0]
            basic_fvc = basic[1]
            basic_percent = basic[2]
            for x in data[uid]:
                week, fvc, percent, age, gender, smoker = x
                self.X.append((basic_week, basic_fvc, basic_percent, week, age, gender, smoker))
                self.y.append(fvc)
                self.uid.append(uid)
        self.X = np.array(self.X)

    @staticmethod
    def transform(array):
        try:
            array = transform_feature.resize(array, (32, 32))
            array = transform_feature.flatten(array)
            return array
        except cv2.error:
            logger.error('error at current ct-scan feature, fill zero instead...')
            return np.zeros((32 * 32, ))

    def _init_ct(self):
        uid_set = set(self.uid)
        self.ct_feature = {}
        for uid in uid_set:
            path_list = loader.fetch_path_by_uid(uid)
            # feature = embedding.embedding_from_files(path_list[:2], transform=self.transform)
            feature = embedding.embedding_from_files(path_list, transform=self.transform)
            self.ct_feature[uid] = feature
        logger.info('ct-scan feature loading finished')

    def get_dataset_with_ct(self, fold=0.9):
        """
        k-fold，带有ct segmentation信息的
        :return:
        """
        self._init_ct()
        X = []
        for i in range(len(self.uid)):
            ct_feature = self.ct_feature[self.uid[i]]
            X.append(embedding.concat(self.X[i], ct_feature))
        X = np.array(X)
        logger.info('finished constructing input X with ct-scan segmentation feature with shape: %s' % str(X.shape))
        X, y = shuffle(self.X, self.y, random_state=RANDOM_STATE)
        size = int(float(fold) / 1. * len(X))
        return np.array(X[:size]), np.array(y[:size]), np.array(X[size:]), np.array(y[size:])

    def get_dataset(self, fold=0.9):
        """
        k-fold cross validation
        :param fold:
        :return:
        """
        X, y = shuffle(self.X, self.y, random_state=RANDOM_STATE)
        X = np.array(X)
        logger.info('finished constructing input X without ct-scan segmentation feature with shape: %s' % str(X.shape))
        size = int(float(fold) / 1. * len(X))
        return np.array(X[:size]), np.array(y[:size]), np.array(X[size:]), np.array(y[size:])

    def get_dataset_with_uid(self, fold=0.9):
        """
        k-fold cross validation
        :param fold:
        :return:
        """
        X, y, uid = shuffle(self.X, self.y, self.uid, random_state=RANDOM_STATE)
        size = int(float(fold) / 1. * len(X))
        return np.array(X[:size]), np.array(y[:size]), uid[:size], np.array(X[size:]), np.array(y[size:]), uid[size:]

    def get_dataset_with_validation(self):
        """
        返回5个测试集的最后三个FVC
        :return:
        """
        pass


if __name__ == '__main__':
    pass
