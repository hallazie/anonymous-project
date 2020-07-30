# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: embedding.py
# @time: 2020/7/20 22:27
# @desc: 将CT图像特征嵌入为向量特征

from config import logger
from ct.mask import masking
from third.lungmask import masking
from ct.transform import transform_feature

import numpy as np
import SimpleITK as sitk


class Embedding:
    def __init__(self):
        self.masking = masking.LungMask()

    def embedding_from_files(self, file_list, feature_raw=None, transform=None):
        image_list = [sitk.GetArrayFromImage(sitk.ReadImage(x))[0] for x in file_list]
        feature = self.embedding(image_list)
        feature = transform_feature.flatten(feature) if not transform else transform(feature)
        if not feature_raw:
            return feature
        assert len(feature_raw.shape) == len(feature.shape)
        feature = self.concat(feature_raw, feature)
        return feature

    def embedding(self, image_list: list) -> np.ndarray:
        # feature_list = np.array([masking.lung_masking(x)[0] for x in image_list])
        feature_list = np.array([self.masking.apply(x) for x in image_list])
        logger.info('current user ct-scan feature loaded with matrix shape: %s' % str(feature_list.shape))
        feature = np.mean(feature_list, axis=0)
        return feature

    @staticmethod
    def concat(v1, v2):
        return np.concatenate((v1, v2), axis=0)


embedding = Embedding()
