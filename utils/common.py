# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: common.py
# @time: 2020/7/12 2:11
# @desc:


def normalize_vector(vector, max_=None, min_=None):
    max_ = max(vector) if max_ is None else max_
    min_ = min(vector) if min_ is None else min_
    vector = list(map(lambda x: float(x - min_) / float(max_ - min_), vector))
    return vector


if __name__ == '__main__':
    pass
