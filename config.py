# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: config.py
# @time: 2020/7/11 15:17
# @desc:

import logging
import math


logger = logging.getLogger('pfp-competition')
logging.basicConfig(level=logging.INFO)


CONST_SQRT2 = math.sqrt(2)

POLYNOMIAL_INTERPOLATION_POWER = 10
RANDOM_STATE = 30

DATA_PATH_ROOT = 'g:\\datasets\\kaggle\\osic-pulmonary-fibrosis-progression'

SYS_PATH_SEP = '/'

logger.info('testing info logger...')
