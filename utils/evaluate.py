# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: evaluate.py
# @time: 2020/7/12 2:59
# @desc: evaluate test result on 5 patient

from utils.criterion import laplace_log_likelihood
from collections import defaultdict

import pandas as pd


def evaluate_on_testset(output):
    test_uid = [row[0] for row in pd.read_csv('data/test.csv').values]
    df = pd.read_csv('data/train.csv')
    header_idx = {k: i for i, k in enumerate(df.columns)}
    data = defaultdict(list)
    for row in df.values:
        uid = row[header_idx['Patient']]
        if uid not in test_uid:
            continue
        week = row[header_idx['Weeks']]
        fvc = row[header_idx['FVC']]
        data[uid].append((week, fvc))
    ground_truth = {}
    for uid in data:
        gt = data[uid][-3:]
        for x in gt:
            ground_truth['%s_%s' % (uid, x[0])] = x[1]
    scores = []
    for x in output:
        uid, fvc, std = x
        if uid not in ground_truth:
            continue
        s = laplace_log_likelihood(ground_truth[uid], fvc, std)
        scores.append(s)
    avg_score = sum(scores) / float(len(scores))
    print('average laplace log-likelihood on test set (%s) = %s' % (len(scores), avg_score))
    return avg_score


if __name__ == '__main__':
    output = {}
    evaluate_on_testset(output)
