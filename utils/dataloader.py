# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: dataloader.py
# @time: 2020/7/14 0:23
# @desc:

from config import RANDOM_STATE, TEST_UID, logger
from ct.embedding import embedding
from ct.loader import loader
from ct.transform import transform_feature
from utils.common import get_abs_path, sample_array_to_bins, matrix_resize

from sklearn.utils import shuffle
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
import pandas as pd
import os
import pydicom
import pickle as pkl


class FVCPredictDataLoader:
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
        for i, uid in enumerate(data):
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

    def _init_ct(self, bins):
        uid_set = set(self.uid)
        self.ct_feature = {}
        for uid in uid_set:
            path_list = loader.fetch_path_by_uid(uid)
            # feature = embedding.embedding_from_files(path_list[:2], transform=self.transform)
            path_list = sample_array_to_bins(path_list, bins)
            feature = embedding.embedding_from_files(path_list, transform=self.transform)
            self.ct_feature[uid] = feature
        logger.info('ct-scan feature loading finished')

    def get_dataset_with_ct(self, bins=10, fold=0.9):
        """
        k-fold，带有ct segmentation信息的
        :return:
        """
        self._init_ct(bins)
        X = []
        for i in range(len(self.uid)):
            ct_feature = self.ct_feature[self.uid[i]]
            X.append(embedding.concat(self.X[i], ct_feature))
        X, y = shuffle(X, self.y, random_state=RANDOM_STATE)
        logger.info('finished constructing input X with ct-scan segmentation feature with shape: %s, %s' % (len(X), str(X[0].shape)))
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
    
    
class AutoEncoderDataLoader:
    def __init__(self):
        super(AutoEncoderDataLoader, self).__init__()


class PolynomialFitRegressionDataset(Dataset):
    def __init__(self, bins=16, train=True, meta_path='checkpoints/polynomial-pow3.json'):
        super(PolynomialFitRegressionDataset, self).__init__()
        self.test_uid = TEST_UID
        self.bins = bins
        self.train = train
        self.meta_path = meta_path
        self.ct_size = (512, 512)
        self.data = defaultdict(list)
        self.label = {}
        self.train_path = get_abs_path(__file__, -2, 'data', 'train.csv')
        self.cache_path = get_abs_path(__file__, -2, 'data', 'ct-scan-cache.pkl')
        self.power = 3
        self._init_meta()
        self._init_label()
        self._init_ct()
        self._init_dataset()

    def _init_meta(self):
        df = pd.read_csv(self.train_path)
        header_idx = {k: i for i, k in enumerate(df.columns)}
        for row in df.values:
            uid = row[header_idx['Patient']]
            if (uid in self.test_uid and self.train) or (uid not in self.test_uid and not self.train):
                continue
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            age = row[header_idx['Age']]
            gender = row[header_idx['Sex']]
            gender_ = 0 if gender == 'Male' else 1
            self.data[uid].append((week, fvc, percent, age, gender_))

    def _init_ct(self):
        self.ct_feature = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                logger.info('using cached ct-feature pickle')
                self.ct_feature = pkl.load(f)
        for i, uid in enumerate(self.data):
            # if i == 10:
            #     break
            if (uid in self.test_uid and self.train) or (uid not in self.test_uid and not self.train):
                continue
            try:
                path_list = loader.fetch_path_by_uid(uid)
                sequence = [pydicom.filereader.dcmread(x) for x in path_list]
                sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
                sequence = sample_array_to_bins(sequence, self.bins, strict=True)
                sequence = [matrix_resize(x.pixel_array.astype('float32'), self.ct_size) for x in sequence]
                seq_size = len(sequence)
                for i in range(self.bins - seq_size):
                    sequence.append(np.zeros(self.ct_size, dtype='float32'))
                # plt.figure(figsize=(20, 12))
                # for i in range(4):
                #     plt.subplot('24%s' % (i*2+1))
                #     plt.imshow(sequence[i*4])
                #     plt.subplot('24%s' % (i*2+2))
                #     plt.imshow(sequence[i*4+1])
                # plt.savefig('../output/bin/%s-plot.png' % uid)
                # print('%s save finished' % uid)
                self.ct_feature[uid] = np.array(sequence)
            except Exception as e:
                logger.error(e)
        with open(self.cache_path, 'wb') as f:
            pkl.dump(self.ct_feature, f)
        logger.info('ct-scan feature loading finished')

    def _init_label(self):
        for uid in self.data:
            if (uid in self.test_uid and self.train) or (uid not in self.test_uid and not self.train):
                continue
            x, y = [e[0] for e in self.data[uid]], [e[1] for e in self.data[uid]]
            z = self._get_fvc_curve_polynomial_coefficient(x, y)
            self.label[uid] = z
        if not self.train:
            return
        c0 = np.array([x[0] for x in self.label.values()])
        c1 = np.array([x[1] for x in self.label.values()])
        c2 = np.array([x[2] for x in self.label.values()])
        c3 = np.array([x[3] for x in self.label.values()])
        self.avg = [np.mean(c0), np.mean(c1), np.mean(c2), np.mean(c3)]
        self.std = [np.std(c0), np.std(c1), np.std(c2), np.std(c3)]
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'avg': self.avg,
                'std': self.std,
                'data-size': len(c0),
            }, f, ensure_ascii=False, indent=4)
        for uid, val in self.label.items():
            self.label[uid] = np.array([
                (val[0] - self.avg[0]) / self.std[0],
                (val[1] - self.avg[1]) / self.std[1],
                (val[2] - self.avg[2]) / self.std[2],
                (val[3] - self.avg[3]) / self.std[3],
            ])

    def _init_dataset(self):
        self.uid = sorted(set(self.ct_feature.keys()) & set(self.label.keys()))
        logger.info('total data size: %s' % len(self.uid))

    def _get_fvc_curve_polynomial_coefficient(self, week_list, fvc_list):
        z = np.polyfit(week_list, fvc_list, self.power)
        return z

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        uid = self.uid[index]
        return self.ct_feature[uid], self.label[uid], uid


if __name__ == '__main__':
    poly_regression = PolynomialFitRegressionDataset()

