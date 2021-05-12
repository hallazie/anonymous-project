# coding:utf-8

import matplotlib.pyplot as plt
import math
import numpy as np

# path = 'log-with-eval3.txt'
# path = 'log'
path = 'log-with-eval-efficientnet.txt'

with open(path, 'r', encoding='utf-8') as f:
    score1, score2 = [], []
    for x in f.readlines():
        if 'final f1 score on test set' not in x:
            continue
        if 'dev final' in x:
            val = x.split('f1 score on test set (546): ')[-1].strip()
            try:
                p = float(val)
                score1.append(p)
            except Exception as e:
                print(e)
        elif 'test final' in x:
            val = x.split('f1 score on test set (546): ')[-1].strip()
            try:
                p = float(val)
                score2.append(p)
            except:
                pass

plt.plot(score1)
plt.plot(score2)
plt.show()








