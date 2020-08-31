# --*-- coding:utf-8 --*--
# @author: xiao shanghua

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
    def base(self, callback):
        if not callable(callback):
            raise TypeError('base should be a callable')
        self._base = callback

    @derive.setter
    def derive(self, drive):
        self._derive = drive

    def build(self, base, derives, coeff_size, expansion_size):
        self.base = base
        self.derive = derives
        self.c_size = coeff_size
        self.e_size = expansion_size
        for i in range(self.e_size):
            self.coeff.append([np.random.rand() for j in range(self.c_size)])

    def forward(self, x):
        try:
            return sum([self.base(self.coeff[i], x) for i in range(self.e_size)])
        except OverflowError as oe:
            print(oe)
            return 0

    def optimize(self, x_list, y_list, iter_num=1000, lr=0.001):
        """
        use least square loss to optimize
        current only support online training, not batch training, or matrix calc
        """
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
                    print(oe)
            print('%sth iter with loss: %s' % ((it + 1), loss))

    def fit(self, source_list):
        return [self.forward(x) for x in source_list]


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
    fitter.build(f, d_list, 3, 128)
    y_list = [0.5825, 0.5571, 0.5186, 0.5395, 0.5206, 0.5287, 0.5033, 0.5193, 0.5176]
    x_list = [i for i in range(len(y_list))]
    z_list = [i / 100. for i in range(len(x_list) * 100)]
    fitter.optimize(x_list, y_list, iter_num=5000, lr=0.01)
    p_list = fitter.fit(z_list)
    plt.plot(x_list, y_list)
    plt.plot(z_list, p_list)
    plt.show()


def test_tanh():
    pass


if __name__ == '__main__':
    test_sigmoid()





