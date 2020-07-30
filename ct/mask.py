# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: mask.py
# @time: 2020/7/12 15:42
# @desc:

from utils.common import normalize_matrix
from third.lungmask import mask
from config import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
import SimpleITK as sitk


class Masking:
    """
    bad case:
        'G:\\datasets\\kaggle\\osic-pulmonary-fibrosis-progression\\train\\ID00026637202179561894768\\19.dcm'
    """
    def __init__(self):
        self.mask_model = mask.get_model('unet', 'LTRCLobes')

    def lung_masking(self, image, vol_postprocessing=True):
        segmentation = mask.apply(image, self.mask_model, volume_postprocessing=vol_postprocessing)
        return segmentation

    def lung_masking_show(self, image):
        segmentation = mask.apply(image, self.mask_model)
        raw = image.pixel_array
        fused = normalize_matrix(segmentation[0]) + normalize_matrix(raw[0])
        return normalize_matrix(fused, 255)

    def lung_masking_from_file(self, file_name):
        image = sitk.ReadImage(file_name)
        segmentation = mask.apply(image, self.mask_model)
        return segmentation

    def lung_masking_from_file_show(self, file_name):
        image = sitk.ReadImage(file_name)
        segmentation = mask.apply(image, self.mask_model)
        raw = sitk.GetArrayFromImage(image)
        fused = normalize_matrix(segmentation[0]) + normalize_matrix(raw[0])
        return normalize_matrix(fused, 255)

    def lung_masking_cv(self, image):
        pass


masking = Masking()


if __name__ == '__main__':
    # source = [-4, 5, 7, 9, 11, 17, 29, 41, 57]
    # target = [2315, 2214, 2061, 2144, 2069, 2101, 2000, 2064, 2057]
    # target_pred, avg = polynomial_interpolation(source, target)
    # logger.info('avg: %s' % avg)
    # plt.scatter(source, target)
    # plt.plot(source, target, label='raw')
    # plt.plot(source, target_pred, label='fit')
    # plt.legend()
    # plt.show()
    pass
