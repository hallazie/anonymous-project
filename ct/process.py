# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: mask.py
# @time: 2020/7/18 1:23
# @desc:

import cv2
import numpy as np


class Processor:
    def __init__(self):
        pass

    @staticmethod
    def edge_detection(image, operator='canny', **kwargs):
        """
        使用canny变换进行边缘检测
        不适用sobel等算子，canny更平滑
        :param image:
        :param operator:
        :return:
        """
        if operator == 'canny':
            thresh1 = kwargs['thresh1'] if 'thresh1' in kwargs else 50
            thresh2 = kwargs['thresh2'] if 'thresh2' in kwargs else 150
            edge = cv2.Canny(image, threshold1=thresh1, threshold2=thresh2)
            edge = cv2.GaussianBlur(edge, (5, 5), 0)
        elif operator == 'sobel':
            edge = cv2.Sobel(image, cv2.CV_8U, 1, 1)
        else:
            edge = cv2.Canny(image)
        return edge

    @staticmethod
    def roi_cropping(image, points):
        """
        将CT画面裁剪，只保留中间圆形区域
        :param image:
        :param points:
        :return:
        """
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [points], 255)
        masked = cv2.bitwise_and(image, mask)
        return masked

    @staticmethod
    def equalizer(image):
        """
        均衡化
        :param image:
        :return:
        """
        eq = cv2.equalizeHist(image)
        return eq


processor = Processor()
