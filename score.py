# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: score.py
# @time: 2020/12/19 17:07
# @desc:

import math


def scoring(date_list, weight_list, resp_list, action_list):
    assert len(date_list) == len(weight_list) == len(resp_list) == len(action_list)
    date_dict = {d: i for i, d in enumerate(set(date_list))}
    p_list = [0 for i in range(len(date_dict))]
    for i, date in enumerate(date_list):
        p_list[date_dict[date]] += (weight_list[i] * resp_list[i] * action_list[i])
    t = sum(p_list) / math.sqrt(sum([p ** 2 for p in p_list])) * math.sqrt(250. / len(p_list))
    u = min(max(t, 0), 6) * sum(p_list)
    return u, t


