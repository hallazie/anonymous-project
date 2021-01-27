# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2020/12/20 14:05
# @desc:

from dataloader import DataLoader
from score import scoring
from mlp_approach import MLP
from regression_approach import REG

import datatable as dt
import json
import torch
import numpy as np


class Tester:
    def __init__(self):
        self.device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
        self.loader = DataLoader()

    def test(self):
        clf = torch.load('checkpoints/mlp-200.pkl')
        pred = clf(torch.from_numpy(self.loader.X_test).float().cuda()).cpu().detach().numpy()

        # pred = np.where(pred[:, 3] > 0.5, 1, 0).astype('int')
        # labl = self.loader.y_test[:, 3]

        # pred = np.where(np.mean(pred, axis=1) > 0, 1, 0).astype('int')
        pred = np.where(pred[:, 0] > 0.5, 1, 0).astype('int')
        labl = self.loader.y_test[:, 0]

        score = scoring(date_list=list(self.loader.d_test), weight_list=list(self.loader.w_test), resp_list=list(self.loader.r_test), action_list=list(pred))
        print(score)

        # labl = np.where(np.mean(self.loader.y_test, axis=1) > 0, 1, 0).astype('int')
        # labl = np.where(np.mean(np.where(self.loader.y_test[0] > 0, 1, 0), axis=1)).astype('int')
        print(self.loader.y_test.shape, pred.shape)

        acc = sum([1 if p == l else 0 for p, l in zip(pred, labl)]) / float(len(pred))
        print(f'pred={pred.shape}, {pred[:3]}..., label={labl.shape}, {labl[:3]}...  -->  score={score}, resp-acc={acc}')

    def run_single(self, ck):
        clf = torch.load(f'checkpoints/mlp-{ck}.pkl')
        pred = clf(torch.from_numpy(self.loader.X_test).float().cuda()).cpu().detach().numpy()
        pred = np.where(pred[:, 0] > 0.5, 1, 0).astype('int')
        labl = self.loader.y_test[:, 0]
        score = scoring(date_list=list(self.loader.d_test), weight_list=list(self.loader.w_test),
                        resp_list=list(self.loader.r_test), action_list=list(pred))
        acc = sum([1 if p == l else 0 for p, l in zip(pred, labl)]) / float(len(pred))
        return score, acc

    def run(self):
        res = {}
        for i in range(1, 25):
            ck = i * 200
            s, a = self.run_single(ck)
            res[ck] = {
                'score': s,
                'acc': a
            }
        with open('output/mlp-5000-result.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    tester = Tester()
    tester.test()



