# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: common.py
# @time: 2020/7/12 2:11
# @desc:

from config import *

import os
import numpy as np
import cv2


def normalize_vector(vector, max_=None, min_=None):
    max_ = max(vector) if max_ is None else max_
    min_ = min(vector) if min_ is None else min_
    vector = list(map(lambda x: float(x - min_) / float(max_ - min_), vector))
    return vector


def normalize_matrix(mat, expand_factor=1):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat)) * expand_factor


def get_abs_path(file_, idx, *args):
    file_list = file_.split(SYS_PATH_SEP)[:-2]
    file_list += args
    return SYS_PATH_SEP.join(file_list)


def sample_array_to_bins(raw_list, bins, strict=False):
    try:
        if len(raw_list) // bins <= 1 and not strict:
            return raw_list
        step = math.ceil(len(raw_list) / float(bins))
        new_list = [x for i, x in enumerate(raw_list) if i % step == 0]
        rest = (len(new_list) - bins) // 2
        if rest > 1:
            return new_list[rest:bins+rest-1]
        else:
            return new_list
    except Exception as e:
        logger.error(e + ' %s %s' % (len(raw_list), bins))
        return raw_list


def matrix_resize(mat, size=(512, 512)):
    return cv2.resize(mat, size, interpolation=cv2.INTER_CUBIC)


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


def polynomial_interpolation(source, target, power=POLYNOMIAL_INTERPOLATION_POWER):
    """
    多项式曲线拟合得到插值结果
    :param source:
    :param target:
    :param power: 插值的次方数
    :return: 插值结果、插值与原值的sqrt
    """
    z = np.polyfit(source, target, power)
    source_ = [i for i in range(min(source), max(source)+1)]
    poly = np.poly1d(z)
    interpolated = poly(source_)
    min_ = min(source)
    avg_ = sum([np.sqrt(np.abs(interpolated[i] - target[i])) for i, x in enumerate(source)]) / float(len(source))
    return interpolated, avg_


if __name__ == '__main__':
    res = get_abs_path('F:/kaggle/pulmonary-fibrosis-progression/utils/dataloader.py', -2, 'data', 'train.csv')
    print(res)