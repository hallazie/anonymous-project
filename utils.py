# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: utils.py
# @time: 2020/12/19 15:45
# @desc:

from config import LOGGER

import time


def monitor_run_time(func):
    def wrap(**kwargs):
        start = time.time()
        ret = func(**kwargs)
        end = time.time()
        LOGGER.info(f'{str(func)} cost {end - start} seconds')
        return ret
    return wrap


if __name__ == '__main__':
    @monitor_run_time
    def foo():
        time.sleep(1)
    foo()
