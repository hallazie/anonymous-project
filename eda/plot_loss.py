# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com

import matplotlib.pyplot as plt


with open('../output/log', 'r', encoding='utf-8') as f:
    loss = []
    for line in f:
        l = float(line.strip().split('loss:')[-1].strip())
        loss.append(l)

plt.plot(loss)
plt.show()

