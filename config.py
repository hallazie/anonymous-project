# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: config.py
# @time: 2020/12/19 15:37
# @desc:

import logging

TRAIN_PATH = 'G:/datasets/kaggle/jane-street/train.csv'

logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
LOGGER = logging.getLogger()

RANDOM_SEED = 30

LAYERS = [256, 384, 512, 512, 512, 768, 768]



