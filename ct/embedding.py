# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: embedding.py
# @time: 2020/7/20 22:27
# @desc: 将CT图像特征嵌入为向量特征

from ct.process import processor

import numpy as np


class Embedding:
    def __init__(self):
        pass

    def embedding(self, image: np.ndarray) -> np.ndarray:
        feature = np.array([])
        return feature

    @staticmethod
    def concat(v1, v2):
        return np.concatenate(v1, v2)
