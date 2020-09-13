# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: common.py
# @time: 2020/7/12 2:11
# @desc:

from config import *
from scipy import interpolate

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
            return new_list[rest:bins + rest - 1]
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


def linear_fill(x, y, target_size=50):
    """
    将x、y填充到target size，x=week list，y=fvc list
    :param x:
    :param y:
    :param target_size:
    :return:
    """

    def closest(e, lst):
        if any([True if (e - ee) == 0 else False for ee in lst]):
            return 0, 0
        cl, cr = lst[0], lst[-1]
        for t in lst:
            if t < e and abs(t-e) <= abs(cl-e):
                cl = t
            if t > e and abs(t-e) <= abs(cr-e):
                cr = t
        return cl, cr

    if len(x) >= target_size:
        return x, y
    data = {k: v for k, v in zip(x, y)}
    step = abs(max(x) - min(x)) / float(target_size)
    fx = [e * step + min(x) for e in range(target_size)]
    pred = {}
    for e in fx:
        xl, xr = closest(e, x)
        if xl == xr:
            continue
        yl, yr = data[xl], data[xr]
        pred[e] = (yr - yl) * ((e - xl) / float(xr - xl)) + yl
    xx = sorted(list(data.keys()) + list(pred.keys()))
    yy = [data[t] if t in data else pred[t] for t in xx]
    return xx, yy


def polynomial_interpolation(source, target, power=POLYNOMIAL_INTERPOLATION_POWER):
    """
    多项式曲线拟合得到插值结果
    :param source:
    :param target:
    :param power: 插值的次方数
    :return: 插值结果、插值与原值的sqrt
    """
    z = np.polyfit(source, target, power)
    source_ = [i for i in range(min(source), max(source) + 1)]
    poly = np.poly1d(z)
    interpolated = poly(source_)
    min_ = min(source)
    avg_ = sum([np.sqrt(np.abs(interpolated[i] - target[i])) for i, x in enumerate(source)]) / float(len(source))
    return interpolated, avg_


def sci_interpolate(source, target, type_='b-spline'):
    """
    基于scipy.interpolate的插值
    :param source:
    :param target:
    :param type_: b-spline, linear
    :return:
    """
    source_fill = [i for i in range(min(source), max(source) + 1)]
    if type_ == 'b-spline':
        tck = interpolate.splrep(source, target)
        return interpolate.splev(source_fill, tck)
    elif type_ == 'linear':
        f_linear = interpolate.interp1d(source, target)
        return f_linear(source_fill)


if __name__ == '__main__':
    res = get_abs_path('F:/kaggle/pulmonary-fibrosis-progression/utils/dataloader.py', -2, 'data', 'train.csv')
    print(res)
    x, y = [0, 1, 3, 5, 7, 13, 26, 37, 52], [0.859875904860393, 0.8839193381592549, 0.915460186142709, 0.905377456049638, 0.8815925542916241, 0.8989141675284391, 0.8451396070320579, 0.8650465356773529, 0.825491209927611]
    xx, yy = linear_fill(x, y, 50)
    print(xx)
    print(yy)


