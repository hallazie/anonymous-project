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

POLYNOMIAL_INTERPOLATION_POWER = 5
RANDOM_STATE = 30

DATA_PATH_ROOT = 'g:\\datasets\\kaggle\\osic-pulmonary-fibrosis-progression'

SYS_PATH_SEP = '/'

TEST_UID = [
            'ID00419637202311204720264',
            'ID00421637202311550012437',
            'ID00422637202311677017371',
            'ID00423637202312137826377',
            'ID00426637202313170790466'
            # 'ID00007637202177411956430',
            # 'ID00009637202177434476278',
            # 'ID00010637202177584971671',
            # 'ID00011637202177653955184',
            # 'ID00012637202177665765362',
        ]

logger.info('testing info logger...')
