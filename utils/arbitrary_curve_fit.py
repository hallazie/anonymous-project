# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from config import logger

import dill
import numpy as np


class ArbitraryCurveFit:
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





