# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from config import logger
from utils.arbitrary_curve_fit_builder import *

import matplotlib.pyplot as plt
import dill
import numpy as np


class AlbitraryCurveFit:
    """
    TODO 添加分段函数性质：y=f(x), x>=0; y=f(x0), x<0
    """
    def __init__(self, _base=None, _derive=None):
        self._base = _base
        self._derive = _derive
        self.e_size = None
        self.c_size = None
        self.coeff = []

    @property
    def base(self):
        return self._base

    @property
    def derive(self):
        return self._derive

    @base.setter
    def base(self, call):
        if not callable(call):
            raise TypeError('base function should be a callable')
        self._base = call

    @derive.setter
    def derive(self, derive):
        if type(derive) is not list or not all([callable(d) for d in derive]):
            raise TypeError('derive function should be a list of callable objs')
        self._derive = derive

    def build(self, base, derives, coeff_size, expansion_size):
        self.base = base
        self.derive = derives
        self.c_size = coeff_size
        self.e_size = expansion_size
        for i in range(self.e_size):
            self.coeff.append([np.random.rand() for j in range(self.c_size)])

    def flush(self):
        self.e_size = None
        self.c_size = None
        self.coeff = []

    def forward(self, x):
        try:
            return sum([self.base(self.coeff[i], x) for i in range(self.e_size)])
        except OverflowError as oe:
            logger.exception(oe)
            return 0

    def optimize(self, x_list, y_list, iter_num=1000, lr=0.001):
        """
        use least square loss to optimize
        current only support online training, not batch training, or matrix calc
        """
        vis_step = iter_num // 10
        for it in range(iter_num):
            loss = 0
            for x, y in zip(x_list, y_list):
                try:
                    p = self.forward(x)
                    d = 2 * (p - y)
                    loss += (p - y) ** 2
                    for e in range(self.e_size):
                        for c in range(self.c_size):
                            self.coeff[e][c] -= self.derive[c](self.coeff[e], x) * lr * d
                except OverflowError as oe:
                    logger.exception(oe)
            if it % vis_step == 0:
                logger.info(f'{it}th iter with loss: {loss}')

    def optimize_batch(self, x_list, y_list, iter_num=1000, lr=0.001, batch_size=10):
        coeff = []
        for i in range(batch_size):
            self.coeff = [[np.random.rand() for j in range(len(self.coeff[0]))] for i in range(len(self.coeff))]
            self.optimize(x_list, y_list, iter_num=iter_num, lr=lr)
            coeff.append(sorted(self.coeff, key=lambda x: [x[j] for j in range(len(x))]))
        self.coeff = list(np.mean(np.array(coeff), axis=0))

    def fit(self, source_list):
        return [self.forward(x) for x in source_list]

    def inspect(self):
        for x in self.coeff:
            logger.info(x)

    def dump(self, path):
        func = {
            'base': self._base,
            'derive': self._derive,
            'coeff': self.coeff
        }
        with open(path, 'wb') as f:
            dill.dump(func, f)

    def load(self, path):
        with open(path, 'rb') as f:
            model = dill.load(f)
            self._base = model.get('base', None)
            self._derive = model.get('derive', None)
            self.coeff = model.get('coeff', None)
            if not callable(self._base):
                raise TypeError('init fitter failed, type confirmation failed')


def build_trigono(size=64):
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
    fitter.build(f, d_list, 6, size)
    return fitter


def build_sigmoid(size=64):
    fitter = AlbitraryCurveFit()
    f = lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -1) * p[2]
    d_list = [
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * x * p[2] * -1,
        lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -2) * np.e ** (p[0] * x + p[1]) * p[2] * -1,
        lambda p, x: (1 + np.e ** (p[0] * x + p[1])) ** -1
    ]
    fitter.build(f, d_list, 3, size)
    return fitter


def build_tanh(size=64):
    fitter = AlbitraryCurveFit()
    f = lambda p, x: p[0] * np.tanh(p[1] * x + p[2])
    d_list = [
        lambda p, x: np.tanh(p[1] * x + p[2]),
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2) * x,
        lambda p, x: p[0] * (1 - np.tanh(p[1] * x + p[2]) ** 2)
    ]
    fitter.build(f, d_list, 3, size)
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





