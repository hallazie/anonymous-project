# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: common.py
# @time: 2020/7/12 2:11
# @desc:

from config import SYS_PATH_SEP

import os


def normalize_vector(vector, max_=None, min_=None):
    max_ = max(vector) if max_ is None else max_
    min_ = min(vector) if min_ is None else min_
    vector = list(map(lambda x: float(x - min_) / float(max_ - min_), vector))
    return vector


def get_abs_path(file_, idx, *args):
    file_list = file_.split(SYS_PATH_SEP)[:-2]
    file_list += args
    return SYS_PATH_SEP.join(file_list)


if __name__ == '__main__':
    res = get_abs_path('F:/kaggle/pulmonary-fibrosis-progression/utils/dataloader.py', -2, 'data', 'train.csv')
    print(res)