# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: loader.py
# @time: 2020/7/18 1:20
# @desc:

from config import logger, DATA_PATH_ROOT, SYS_PATH_SEP
from ct.process import processor
from utils.common import normalize_matrix

# import matplotlib.pyplot as plt
import pydicom
import numpy as np
import os


class Loader:
    def __init__(self):
        self.train_data = {}
        self._fetch_patient_files()

    def _fetch_patient_files(self):
        self.train_user_dirs, self.test_user_dirs = [], []
        self.train_user_dict, self.test_user_dict = {}, {}
        self.train_uid, self.test_uid = [], []
        for d in os.listdir(os.path.join(DATA_PATH_ROOT, 'train')):
            path = os.path.join(DATA_PATH_ROOT, 'train', d)
            if not os.path.isdir(path):
                continue
            self.train_user_dirs.append(path)
            self.train_user_dict[d] = path
            self.train_uid.append(d)
        for d in os.listdir(os.path.join(DATA_PATH_ROOT, 'test')):
            path = os.path.join(DATA_PATH_ROOT, 'test', d)
            if not os.path.isdir(path):
                continue
            self.test_user_dirs.append(path)
            self.test_uid.append(d)
            self.test_user_dict[d] = path

    def _fetch_data(self):
        for idx, f in enumerate(self.train_user_dirs):
            try:
                f_list = os.listdir(f)
                sequence = [pydicom.filereader.dcmread(os.path.join(f, x)) for x in f_list]
                sequence = sorted(sequence, key=lambda x: x.ImagePositionPatient[2])
                self.train_data[f.split('\\')[-1]] = sequence
            except ValueError as ve:
                logger.error('%s failed with value error: %s' % (f, ve))
            except AttributeError as ae:
                logger.error('%s failed with attribute error: %s' % (f, ae))
            finally:
                logger.info('user data: %s loaded' % f.split('\\')[-1])
        logger.info('fetching data finished with size: %s' % len(self.train_data))

    def fetch_array_by_uid(self, uid):
        return self.train_data[uid]

    def fetch_path_by_uid(self, uid):
        if uid in self.train_user_dict:
            root = self.train_user_dict[uid]
            ret = []
            for f in os.listdir(root):
                if f.endswith('.dcm'):
                    ret.append(os.path.join(root, f))
            return ret
        elif uid in self.test_user_dict:
            root = self.test_user_dict[uid]
            ret = []
            for f in os.listdir(root):
                if f.endswith('.dcm'):
                    ret.append(os.path.join(root, f))
            return ret

    def visualize(self):
        self._fetch_data()
        import random
        uid = 'ID00010637202177584971671'
        image_list = self.fetch_array_by_uid(uid)
        image = image_list[random.randint(0, len(image_list)-1)].pixel_array
        image = normalize_matrix(image, expand_factor=255)
        image = np.uint8(image)
        # plt.subplot('121')
        # plt.imshow(image)
        # plt.subplot('122')
        # plt.imshow(processor.edge_detection(image, 'canny', thresh1=100, thresh2=200))
        # plt.show()


loader = Loader()

if __name__ == '__main__':
    loader.visualize()
