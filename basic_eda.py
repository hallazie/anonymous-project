# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: basic_eda.py
# @time: 2020/12/19 15:36
# @desc:

from config import *
from utils import *

import datatable as dt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def batch_histogram():
    df = dt.fread(TRAIN_PATH)
    df = df.to_pandas()
    for i, name in enumerate(df.columns):
        col = df[name]
        plt.hist(col, bins=50)
        plt.savefig(f'output/figs/basic-eda/hist/{i}-{name}.png')
        plt.clf()
        print(f'{i}th {name} save finished...')


def accum_plot():
    df = dt.fread(TRAIN_PATH)
    df = df.to_pandas()
    for name in ['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']:
        col = df[name]
        cur, accum = 0, []
        for val in col:
            cur += val
            accum.append(cur)
        plt.plot(accum, label=name)
    plt.legend()
    plt.savefig(f'output/figs/basic-eda/accum-resp-total.png', dpi=256)
    plt.clf()


def step_plot():
    df = dt.fread(TRAIN_PATH)
    df = df.to_pandas()
    for name in ['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']:
        col = df[name]
        plt.plot(col, label=name)
        std = np.std(col)
        mean = np.mean(col)
        plt.legend()
        plt.savefig(f'output/figs/basic-eda/step-{name}-{mean}-{std}.png', dpi=256)
        plt.clf()


def plot_weight():
    df = dt.fread(TRAIN_PATH)
    df = df.to_pandas()
    col = df['weight']
    raw_size = len(col)
    col = [x for x in col if x != 0 and x < 1.5]
    new_size = len(col)
    print(f'raw size: {raw_size}, new size: {new_size}')
    plt.figure(dpi=256)
    plt.hist(col, bins=512)
    plt.show()
    # plt.savefig(f'output/figs/basic-eda/weight-hist-trunc-2.png')
    # plt.clf()


if __name__ == '__main__':
    plot_weight()
