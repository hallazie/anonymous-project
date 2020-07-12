# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: process.py
# @time: 2020/7/12 15:42
# @desc:

from utils.common import normalize_vector
from config import *

import matplotlib.pyplot as plt
import numpy as np


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


def polynomial_interpolation(source, target, power=CONST_POLYNOMIAL_INTERPOLATION_POWER):
    """
    多项式曲线拟合得到插值结果
    :param source:
    :param target:
    :param power: 插值的次方数
    :return: 插值结果、插值与原值的sqrt
    """
    poly = np.poly1d(np.polyfit(source, target, power))
    interpolated = poly(source)
    min_ = min(source)
    avg_ = sum([np.sqrt(interpolated[x - min_] - target[i]) for i, x in enumerate(source)]) / float(len(source))
    return interpolated, avg_


if __name__ == '__main__':
    source = [-4, 5, 7, 9, 11, 17, 29, 41, 57]
    target = [2315, 2214, 2061, 2144, 2069, 2101, 2000, 2064, 2057]
    z1 = np.polyfit(source, target, 8)
    p1 = np.poly1d(z1)
    target_pred = p1(source)
    plt.scatter(source, target)
    plt.plot(source, target, label='raw')
    plt.plot(source, target_pred, label='fit')
    plt.legend()
    plt.show()
    # source_ = linear_interpolation(source, source)
    # target_ = linear_interpolation(source, target)
    # plt.subplot(211)
    # plt.plot(normalize_vector(target))
    # plt.plot(normalize_vector(source))
    # plt.subplot(212)
    # plt.plot(normalize_vector(target_))
    # plt.plot(normalize_vector(source_))
    # plt.show()
