# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: arbitraryfit_loader.py
# @time: 2020/9/13 19:43
# @desc:

from config import RANDOM_STATE, TEST_UID, logger
from ct.loader import loader
from utils.common import get_abs_path, sample_array_to_bins, matrix_resize
from utils.arbitrary_curve_fit_builder import build_sigmoid
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

import json
import numpy as np
import pandas as pd
import os
import pydicom
import pickle as pkl


class ArbitraryFitRegressionDataset(Dataset):
    def __init__(self, bins=16, train=True):
        super(ArbitraryFitRegressionDataset, self).__init__()
        self.test_uid = TEST_UID
        self.bins = bins                # 对CT scan进行分桶
        self.train = train
        self.ct_size = (512, 512)
        self.data = defaultdict(list)
        self.label = {}
        self.arbitrary_model_dict = {}
        self.train_path = get_abs_path(__file__, -2, 'data', 'train.csv')
        self.cache_path = get_abs_path(__file__, -2, 'data', 'ct-scan-cache.pkl')
        self.fitter_path = get_abs_path(__file__, 2, 'output', 'model', 'checkpoint', '0913')
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
            sigmoid = build_sigmoid(32)
            sigmoid.load(os.path.join(self.fitter_path, f'{uid}.pkl'))
            coeff = sorted(sigmoid.coeff, key=lambda x: sum([y**2 for y in x]))
            self.label[uid] = [y for x in coeff for y in x]
            print(self.label[uid])

    def _init_dataset(self):
        self.uid = sorted(set(self.ct_feature.keys()) & set(self.label.keys()))
        logger.info('total data size: %s' % len(self.uid))

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        uid = self.uid[index]
        return self.ct_feature[uid], self.label[uid], uid


if __name__ == '__main__':
    loader = ArbitraryFitRegressionDataset()


