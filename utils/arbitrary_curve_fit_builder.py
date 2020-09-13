# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from utils.arbitrary_curve_fit import ArbitraryCurveFit

import numpy as np
import matplotlib.pyplot as plt


def build_trigono(size=32):
    fitter = ArbitraryCurveFit()
    f = lambda p, x: p[0] * np.sin(p[1] * x + p[2]) + p[3] * np.cos(p[4] * x + p[5])
    d_list = [
        lambda p, x: np.sin(p[1] * x + p[2]),
        lambda p, x: p[0] * np.cos(p[1] * x + p[2]) * x,
        lambda p, x: p[0] * np.cos(p[1] * x + p[2]),
        lambda p, x: np.cos(p[4] * x + p[5]),
        lambda p, x: p[3] * np.sin(p[4] * x + p[5]) * x * -1,
        lambda p, x: p[3] * np.sin(p[4] * x + p[5]) * -1
    ]
    fitter.build(f, d_list, len(d_list), size)
    return fitter


def build_sigmoid(size=32):
    fitter = ArbitraryCurveFit()
    f = lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -1) * p[2]
    d_list = [
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * x * p[2] * -1,
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * p[2] * -1,
        lambda p, x: (1 + np.e ** (p[0] * x + p[1])) ** -1
    ]
    fitter.build(f, d_list, len(d_list), size)
    return fitter


def build_tanh(size=32):
    fitter = ArbitraryCurveFit()
    f = lambda p, x: p[0] * np.tanh(p[1] * x + p[2])
    d_list = [
        lambda p, x: np.tanh(p[1] * x + p[2]),
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2) * x,
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2)
    ]
    fitter.build(f, d_list, len(d_list), size)
    return fitter


def build_gaussian(size=32):
    fitter = ArbitraryCurveFit()
    f = lambda p, x: p[0] * np.e ** ((-1 * (x - p[1]) ** 2) / (2 * p[2] ** 2)) + p[3]
    d_list = [
        lambda p, x: np.e ** ((-1 * (x - p[1]) ** 2) / (2 * p[2] ** 2)),
        lambda p, x: p[0] * np.e ** ((-1 * (x - p[1]) ** 2) / (2 * p[2] ** 2)) * ((x - p[1]) / p[2]),
        lambda p, x: p[0] * np.e ** ((-1 * (x - p[1]) ** 2) / (2 * p[2] ** 2)) * ((x - p[1]) ** 2 / p[2] ** 3),
        lambda p, x: 1
    ]
    fitter.build(f, d_list, len(d_list), size)
    return fitter


def test():
    fitter = build_sigmoid()

    # y_list = [65.3061224489796, 58.1632653061225, 57.8656462585034, 62.2874149659864, 71.4710884353742, 62.1173469387755, 56.8452380952381, 55.4421768707483, 59.7789115646259]
    # x_list = [-3, 3, 5, 7, 9, 15, 27, 39, 55]
    # y_list = [e/100. for e in y_list]
    # x_list = [e-min(x_list) for e in x_list]

    y_list = [0.859875904860393, 0.8839193381592549, 0.915460186142709, 0.905377456049638, 0.8815925542916241, 0.8989141675284391, 0.8451396070320579, 0.8650465356773529, 0.825491209927611]
    x_list = [0, 1, 3, 5, 7, 13, 26, 37, 52]

    z_list = [i / 100. for i in range(min(x_list) * 100, (max(x_list) + 1) * 200)]
    for i in range(10):
        fitter.optimize_batch(x_list, y_list, iter_num=1000, lr=0.005)
        fitter.inspect()
        p_list = fitter.fit(z_list)
        plt.plot(x_list, y_list)
        plt.plot(z_list, p_list)
        # plt.show()
        plt.savefig('../output/train/%s.jpg' % i)
        plt.clf()


if __name__ == '__main__':
    test()
