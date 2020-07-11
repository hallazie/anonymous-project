# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: criterion.py
# @time: 2020/7/11 15:09
# @desc:

import math
from config import *


def laplace_log_likelihood(label_fvc, predict_fvc, sigma):
    sigma_clip = max(sigma, 70)
    delta = min(abs(label_fvc - predict_fvc), 1000)
    metric = - (CONST_SQRT2 * delta / sigma_clip) - math.log(CONST_SQRT2 * sigma_clip)
    return metric

