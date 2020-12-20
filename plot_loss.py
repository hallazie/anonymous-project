# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: plot_loss.py
# @time: 2020/12/20 15:36
# @desc:

import matplotlib.pyplot as plt


with open('log', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    loss = []
    for x in lines:
        try:
            loss.append(float(x.split('loss=')[-1].strip()))
        except Exception as e:
            print(f'line: {x} failed with error: {e}')
    plt.plot(loss)
    plt.show()
