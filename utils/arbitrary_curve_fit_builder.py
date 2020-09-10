# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from utils.arbitrary_curve_fit import AlbitraryCurveFit

import numpy as np


def build_trigono():
    fitter = AlbitraryCurveFit()
    f = lambda p, x: p[0] * np.sin(p[1] * x + p[2]) + p[3] * np.cos(p[4] * x + p[5])
    d_list = [
        lambda p, x: np.sin(p[1] * x + p[2]),
        lambda p, x: p[0] * np.cos(p[1] * x + p[2]) * x,
        lambda p, x: p[0] * np.cos(p[1] * x + p[2]),
        lambda p, x: np.cos(p[4] * x + p[5]),
        lambda p, x: p[3] * np.sin(p[4] * x + p[5]) * x * -1,
        lambda p, x: p[3] * np.sin(p[4] * x + p[5]) * -1
    ]
    fitter.build(f, d_list, 6, 20)
    return fitter


def build_sigmoid():
    fitter = AlbitraryCurveFit()
    f = lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -1) * p[2]
    d_list = [
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * x * p[2] * -1,
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * p[2] * -1,
        lambda p, x: (1 + np.e ** (p[0] * x + p[1])) ** -1
    ]
    fitter.build(f, d_list, 3, 32)
    return fitter


def build_tanh():
    fitter = AlbitraryCurveFit()
    f = lambda p, x: p[0] * np.tanh(p[1] * x + p[2])
    d_list = [
        lambda p, x: np.tanh(p[1] * x + p[2]),
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2) * x,
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2)
    ]
    fitter.build(f, d_list, 3, 128)
    return fitter

