# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: arbitraryfit_loader.py
# @time: 2020/9/13 19:43
# @desc:

from config import RANDOM_STATE, TEST_UID, logger
from ct.loader import loader
from utils.common import get_abs_path, sample_array_to_bins, matrix_resize

from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

import json
import numpy as np
import pandas as pd
import os
import pydicom
import pickle as pkl


class ArbitraryFitRegressionDataset(Dataset):
    def __init__(self, bins=16, train=True, meta_path='checkpoints/polynomial-pow3.json'):
        super(ArbitraryFitRegressionDataset, self).__init__()
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
