# --*-- coding:utf-8 --*--
# @author: xiao shanghua

from collections import defaultdict
from config import *
from utils import *

import pandas as pd
import matplotlib.pyplot as plt


def norm(array):
    max_, min_ = max(array), min(array)
    # array = map(lambda x: ((x - min_) / float(max_ - min_)) if max_ != min_ else 0, array)
    array = [((x - min_) / float(max_ - min_)) if max_ != min_ else 0 for x in array]
    return list(array)


df = pd.read_csv('data/train.csv').fillna('')
data = {x[0]: {'text': x[3], 'target': x[4], 'std': x[5]} for x in df.values}
token_counter = defaultdict(int)
for item in data.values():
    for s in segment(item['text']):
        token_counter[s] += 1

ease_list, freq_list = [], []
for k, item in data.items():
    freq = []
    for s in segment(item['text']):
        if s not in token_counter:
            continue
        freq.append(token_counter[s])
    avg = sum(freq) / float(len(freq))
    ease_list.append(item['target'])
    freq_list.append(avg)

ease_list = norm(ease_list)
freq_list = norm(freq_list)

plt.subplot(211)
plt.plot(ease_list)
plt.subplot(212)
plt.plot(freq_list)
plt.show()












