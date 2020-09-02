# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from config import logger

import matplotlib.pyplot as plt
import numpy as np


class AlbitraryCurveFit:
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

    def fit(self, source_list):
        return [self.forward(x) for x in source_list]

    def inspect(self):
        for x in self.coeff:
            logger.info(x)


def test_trigono():
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
    y_list = [0.5825, 0.5571, 0.5186, 0.5395, 0.5206, 0.5287, 0.5033, 0.5193, 0.5176]
    x_list = [i for i in range(len(y_list))]
    z_list = [i / 100. for i in range(len(x_list) * 100)]
    fitter.optimize(x_list, y_list, lr=0.001)
    fitter.inspect()
    p_list = fitter.fit(z_list)
    plt.plot(x_list, y_list)
    plt.plot(z_list, p_list)
    plt.show()


def test_sigmoid():
    fitter = AlbitraryCurveFit()
    f = lambda p, x: ((1 + np.e ** (p[0] * x + p[1])) ** -1) * p[2]
    d_list = [
        lambda p, x: ((1 + np.e ** (p[0] * x)) ** -2) * x * p[2] * -1,
        lambda p, x: ((1 + np.e ** (p[0] * x)) ** -2) * p[2] * -1,
        lambda p, x: (1 + np.e ** (p[0] * x)) ** -1
    ]
    fitter.build(f, d_list, 3, 256)
    # y_list = [0.5825, 0.5571, 0.5186, 0.5395, 0.5206, 0.5287, 0.5033, 0.5193, 0.5176]
    # x_list = [i for i in range(len(y_list))]
    # z_list = [i / 100. for i in range(len(x_list) * 100)]
    y_list = [0.859875904860393, 0.8839193381592549, 0.915460186142709, 0.905377456049638, 0.8815925542916241, 0.8989141675284391, 0.8451396070320579, 0.8650465356773529, 0.825491209927611]
    x_list = [0, 1, 3, 5, 7, 13, 26, 37, 52]
    z_list = [i / 100. for i in range(min(x_list) * 100, (max(x_list) + 1) * 100)]
    fitter.optimize(x_list, y_list, iter_num=1000, lr=0.01)
    fitter.inspect()
    p_list = fitter.fit(z_list)
    plt.plot(x_list, y_list)
    plt.plot(z_list, p_list)
    plt.show()


def test_tanh():
    pass


if __name__ == '__main__':
    test_sigmoid()





