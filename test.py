# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2020/12/20 14:05
# @desc:

from sklearn.model_selection import train_test_split
from mlp_model import MLPModel
from config import *
from score import scoring

import datatable as dt
import torch
import numpy as np


class Tester:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.epoch = 4096
        self.batch_size = 4096
        self.lr = 1e-3

    def test(self):
        model = MLPModel(130, LAYERS).to(self.device)
        model.eval()
        checkpoint = torch.load('checkpoints/baseline.pkl')
        model.load_state_dict(state_dict=checkpoint, strict=True)

        train = dt.fread(TRAIN_PATH)
        train = train.to_pandas()
        train = train[train['weight'] != 0]
        train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

        feature = [False] * 7 + [True] * 130 + [False] * 2
        # x_train = train.loc[:, train.columns.str.contains('feature')]
        x_train = train.loc[:, feature]
        y_train = train.loc[:, 'action']
        w_train = train.loc[:, 'weight']
        r_train = train.loc[:, 'resp']
        d_train = train.loc[:, 'date']

        train_x, test_x, train_y, test_y, train_w, test_w, train_r, test_r, train_d, test_d = train_test_split(x_train, y_train, w_train, r_train, d_train, random_state=666, test_size=0.1)
        train_x = train_x.fillna(0)
        test_x = test_x.fillna(0)
        print(test_x.shape)
        pred = model(torch.from_numpy(np.array(test_x)).to(self.device).float())[:, 0].cpu().detach().numpy()
        print(pred.shape)
        score = scoring(date_list=list(test_d), weight_list=list(test_w), resp_list=list(test_r), action_list=list(pred))
        print(score)


if __name__ == '__main__':
    tester = Tester()
    tester.test()



