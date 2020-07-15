# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: process.py
# @time: 2020/7/12 15:42
# @desc:

from utils.common import normalize_matrix
from third.lungmask import mask
from config import *

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


class Processor:
    def __init__(self):
        self.mask_model = mask.get_model('unet','LTRCLobes')

    @staticmethod
    def linear_interpolation(source, target):
        """
        对FVC等sequence线性插值，填满weeks对应的数据
        :param source:
        :param target:
        :return: 插值后的FVC list
        """
        if len(source) < 2:
            return target
        interpolated = []
        min_, max_ = min(source), max(source)
        for i in range(len(source) - 1):
            h, t = source[i], source[i + 1]
            interpolated.append(target[i])
            step = (target[i + 1] - target[i]) / float(t - h)
            for j in range(t - h):
                interpolated.append(step * (j + 1) + target[i])
        return interpolated

    @staticmethod
    def polynomial_interpolation(source, target, power=POLYNOMIAL_INTERPOLATION_POWER):
        """
        多项式曲线拟合得到插值结果
        :param source:
        :param target:
        :param power: 插值的次方数
        :return: 插值结果、插值与原值的sqrt
        """
        z = np.polyfit(source, target, power)
        poly = np.poly1d(z)
        interpolated = poly(source)
        min_ = min(source)
        avg_ = sum([np.sqrt(np.abs(interpolated[i] - target[i])) for i, x in enumerate(source)]) / float(len(source))
        return interpolated, avg_

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


processor = Processor()


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
