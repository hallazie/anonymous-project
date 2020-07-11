# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: image.py
# @time: 2020/7/11 21:57
# @desc:

import numpy as np
import cv2


def normalize_image(array, size=None):
    if type(array) is np.ndarray:
        array = (array - np.min(array)) / (np.max(array) - np.min(array))
    if size:
        array = cv2.resize(array, size, interpolation=cv2.INTER_CUBIC)
    return array


