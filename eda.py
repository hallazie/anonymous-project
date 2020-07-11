# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: eda.py
# @time: 2020/7/11 15:09
# @desc:

from config import logger, DATA_PATH_ROOT
from utils.image import normalize_image

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
import seaborn as sns
import imageio
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
        dset = pydicom.filereader.dcmread(path)
        print(dset)
        plt.figure(figsize=(5, 5))
        plt.grid(False)
        plt.imshow(dset.pixel_array, cmap='gray')
        plt.show()

    def _visualize_random_sequence(self):
        d_idx = random.randint(0, len(self.train_user_dirs))
        f_list = os.listdir(self.train_user_dirs[d_idx])
        sequence = [pydicom.filereader.dcmread(os.path.join(self.train_user_dirs[d_idx], x)) for x in f_list]
        sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
        sequence = [normalize_image(x.pixel_array, (512, 512)) for x in sequence]
        imageio.mimsave('output/visualization/random-sequence.gif', sequence, duration=0.0001)
    
    def _visualize_all_sequence(self):
        for f in self.train_user_dirs:
            try:
                idx = f.split(os.sep)[-1]
                f_list = os.listdir(f)
                sequence = [pydicom.filereader.dcmread(os.path.join(f, x)) for x in f_list]
                sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
                sequence = [normalize_image(x.pixel_array, (512, 512)) for x in sequence]
                imageio.mimsave('output/visualization/%s.gif' % idx, sequence, duration=0.0001)
                logger.info('%s finished!' % f)
            except ValueError as ve:
                logger.error('%s failed with value error: %s' % (f, ve))
            except AttributeError as ae:
                logger.error('%s failed with attribute error: %s' % (f, ae))
            except Exception as e:
                logger.error('%s failed with other error: %s' % (f, e))

    def run(self):
        self._visualize_all_sequence()


if __name__ == '__main__':
    eda = EDA()
    eda.run()
