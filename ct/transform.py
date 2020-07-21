# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: transform.py
# @time: 2020/7/22 0:27
# @desc:

import numpy as np
import cv2


class TransformFeatureFromLungSegment:
    def __init__(self):
        pass

    @staticmethod
    def flatten(array: np.ndarray):
        return array.flatten()

    @staticmethod
    def resize(array: np.ndarray, shape: tuple):
        return cv2.resize(array, dsize=shape, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def transform(array: np.ndarray, *funcs):
        for func in funcs:
            array = func(array)
        return array


transform_feature = TransformFeatureFromLungSegment()
