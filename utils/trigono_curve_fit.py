# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: trigono_curve_fit.py
# @time: 2020/8/27 23:34
# @desc:

import matplotlib.pyplot as plt

import numpy as np


class TrigonometricCurveFit:
    """
    三角函数拟合曲线
    """
    def __init__(self):
        self.size = None
        self.ws1 = None
        self.ws2 = None
        self.bs1 = None
        self.wc1 = None
        self.wc2 = None
        self.bc1 = None

    def fit(self, source):
        """
        返回拟合后的曲线
        :return:
        """
        return [self._tri(x) for x in source]

    def build(self, pi_list):
        """
        构建表达式，pi list=[1, 2, 3, ... 1/2, 1/4, ...] etc
        :param pi_list:
        :return:
        """
        self.size = len(pi_list)
        self.ws1 = [np.random.rand() for i in range(self.size)]
        self.ws2 = [np.random.rand() for i in range(self.size)]
        self.bs1 = [np.random.rand() for i in range(self.size)]
        self.wc1 = [np.random.rand() for i in range(self.size)]
        self.wc2 = [np.random.rand() for i in range(self.size)]
        self.bc1 = [np.random.rand() for i in range(self.size)]

    def optimize(self, x_list, y_list, iter_nums=100, lr=0.005):
        """
        使用最小二乘法优化
        :return:
        """
        for it in range(iter_nums):
            for x, y in zip(x_list, y_list):
                p = self._tri(x)
                print('current loss: %s' % ((p - y) ** 2))
                d = 2 * (p - y)
                d_ws1 = [np.cos(self.ws1[i] * x + self.bs1[i]) for i in range(self.size)]
                d_ws2 = [self.ws2[i] * x * np.sin(self.ws1[i] * x + self.bs1[i]) for i in range(self.size)]
                d_bs1 = [self.ws2[i] * np.sin(self.ws1[i] * x + self.bs1[i]) for i in range(self.size)]
                d_wc1 = [np.cos(self.wc1[i] * x + self.bc1[i]) for i in range(self.size)]
                d_wc2 = [self.wc2[i] * x * np.sin(self.wc1[i] * x + self.bc1[i]) for i in range(self.size)]
                d_bc1 = [self.wc2[i] * np.sin(self.wc1[i] * x + self.bc1[i]) for i in range(self.size)]
                self.ws1 = [self.ws1[i] - lr * d * d_ws1[i] for i in range(self.size)]
                self.ws2 = [self.ws2[i] - lr * d * d_ws2[i] for i in range(self.size)]
                self.bs1 = [self.bs1[i] - lr * d * d_bs1[i] for i in range(self.size)]
                self.wc1 = [self.wc1[i] - lr * d * d_wc1[i] for i in range(self.size)]
                self.wc2 = [self.wc2[i] - lr * d * d_wc2[i] for i in range(self.size)]
                self.bc1 = [self.bc1[i] - lr * d * d_bc1[i] for i in range(self.size)]

    def _tri(self, x):
        y = sum([self.ws2[i] * np.sin(self.ws1[i] * x + self.bs1[i]) + self.wc2[i] * np.cos(self.wc1[i] * x + self.bc1[i]) for i in range(self.size)])
        return y


if __name__ == '__main__':
    tri = TrigonometricCurveFit()
    par = [1./i for i in range(1, 10)]
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 2, 1, -2, 0, 1.5]
    s = [i/100. for i in range(500)]
    tri.build(par)
    tri.optimize(x, y, iter_nums=1000, lr=0.001)
    t = tri.fit(s)
    plt.plot(x, y)
    plt.plot(s, t)
    plt.show()



