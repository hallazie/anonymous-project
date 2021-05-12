# coding:utf-8

import matplotlib.pyplot as plt
import math
import numpy as np

# path = 'log-with-eval3.txt'
# path = 'log'
path = 'log-with-eval-efficientnet-b5.txt'

with open(path, 'r', encoding='utf-8') as f:
    loss = []
    for x in f.readlines():
        if 'loss=' not in x:
            continue
        val = x.split('loss=')[-1].strip()
        try:
            p = float(val)
            loss.append(p)
        except:
            pass

print(len(loss))
plt.plot(loss)
plt.show()


from efficientnet_pytorch import EfficientNet

# model = EfficientNet.from_pretrained('efficientnet-b0')
# print(model)




