# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: eda.py
# @time: 2020/7/11 15:09
# @desc:

from config import logger, DATA_PATH_ROOT
from ct.process import processor
from ct.mask import masking
from utils.image import normalize_image
from utils.common import normalize_vector, polynomial_interpolation, linear_interpolation
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
import imageio
import random
import traceback
import os
import SimpleITK as sitk


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

    def _visualize_random_single(self):
        d_idx = random.randint(0, len(self.train_user_dirs))
        f_list = os.listdir(self.train_user_dirs[d_idx])
        f_idx = random.randint(0, len(f_list))
        path = os.path.join(self.train_user_dirs[d_idx], f_list[f_idx])
        path = 'G:\\datasets\\kaggle\\osic-pulmonary-fibrosis-progression\\train\\ID00130637202220059448013\\19.dcm'
        dset = pydicom.filereader.dcmread(path)
        mask = masking.lung_masking(sitk.ReadImage(path), vol_postprocessing=True)[0]
        print(dset)
        # plt.figure(figsize=(5, 5))
        # plt.grid(False)
        plt.subplot(121)
        plt.imshow(dset.pixel_array, cmap='gray')
        plt.subplot(122)
        plt.imshow(mask)
        plt.show()

    def _visualize_random_sequence(self):
        d_idx = random.randint(0, len(self.train_user_dirs))
        f_list = os.listdir(self.train_user_dirs[d_idx])
        sequence = [pydicom.filereader.dcmread(os.path.join(self.train_user_dirs[d_idx], x)) for x in f_list]
        sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
        sequence = [normalize_image(x.pixel_array, (512, 512)) for x in sequence]
        imageio.mimsave('output/visualization/random-sequence.gif', sequence, duration=0.0001)
    
    def _visualize_all_sequence(self):
        for idx, f in enumerate(self.train_user_dirs):
            try:
                idx = f.split(os.sep)[-1]
                f_list = os.listdir(f)
                # sequence = [pydicom.filereader.dcmread(os.path.join(f, x)) for x in f_list]
                sequence = [processor.lung_masking_from_file_show(os.path.join(f, x)) for x in f_list]
                # sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
                # sequence = [processor.lung_masking_show(x) for x in sequence]
                imageio.mimsave('output/segmentation/%s.gif' % idx, sequence, duration=0.0001)
                logger.info('%s finished!' % f)
            except ValueError as ve:
                logger.error('%s failed with value error: %s' % (f, ve))
            except AttributeError as ae:
                logger.error('%s failed with attribute error: %s' % (f, ae))
                traceback.print_exc()
            except Exception as e:
                logger.error('%s failed with other error: %s' % (f, e))
                traceback.print_exc()

    @staticmethod
    def _plot_all_sequence():
        df = pd.read_csv('data/train.csv')
        header_idx = {k: i for i, k in enumerate(df.columns)}
        plots = {}
        for row in df.values:
            uid = row[header_idx['Patient']]
            week = row[header_idx['Weeks']]
            fvc = row[header_idx['FVC']]
            percent = row[header_idx['Percent']]
            if uid not in plots:
                plots[uid] = {'week': [], 'fvc': [], 'percent': []}
            plots[uid]['week'].append(week)
            plots[uid]['fvc'].append(fvc)
            plots[uid]['percent'].append(percent)
        for uid in plots:
            fig = plt.figure()
            week_ = normalize_vector(plots[uid]['week'])
            fvc_ = normalize_vector(plots[uid]['fvc'], min_=0)
            percent_ = [x / 100. for x in plots[uid]['percent']]
            percent_norm = normalize_vector(plots[uid]['percent'])
            idx_ = [i for i in range(len(week_))]
            plt.plot(idx_, week_, label='week')
            plt.plot(idx_, fvc_, label='fvc')
            plt.plot(idx_, percent_, label='percent')
            plt.plot(idx_, percent_norm, label='percent-normed')
            plt.legend()
            plt.savefig('output/segmentation/%s-plot.png' % uid)
            plt.clf()
            print('%s saved: %s, %s, %s' % (uid, str(week_[:2]), str(fvc_[:2]), str(percent_[:2])))

    def _fit_all_fvc_curve(self):
        for uid in self.data:
            fvc_list = [x[1] for x in self.data[uid]]
            week_list = [x[0] for x in self.data[uid]]
            fvc_list_interpolated, div = polynomial_interpolation(week_list, fvc_list, power=3)
            week_list_ = [i for i in range(min(week_list), max(week_list)+1)]
            plt.plot(week_list, fvc_list)
            plt.plot(week_list_, fvc_list_interpolated)
            plt.savefig('output/fvc-curve/%s-plot-pow3.png' % uid)
            print(uid, 'polynomial interpolation finished')
            plt.clf()

    def run(self):
        # self._visualize_all_sequence()
        # self._plot_all_sequence()
        self._fit_all_fvc_curve()


if __name__ == '__main__':
    eda = EDA()
    eda.run()
