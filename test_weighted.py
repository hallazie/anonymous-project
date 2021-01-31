# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: train.py
# @time: 2020/12/20 14:05
# @desc:

from dataloader import DataLoader
# from dataloader_spand import DataLoader
from score import scoring, utility_score_bincount
from sklearn.metrics import roc_curve, auc as area_under_roc
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
        clf = torch.load('checkpoints/mlp-weighted-200.pkl')
        # clf = torch.load('checkpoints/mlp-10000.pkl')
        clf.eval()
        for param in clf.parameters():
            param.requires_grad = False

        pred = clf(torch.from_numpy(self.loader.X_test).float().cuda()).cpu().detach().numpy()
        weit = np.tile(np.clip(self.loader.w_test.values, a_min=0.25, a_max=8), (5, 1)).T
        pred = pred / weit
        pred = np.where(np.mean(np.where(pred > 0.5, 1, 0), axis=1) > 0.5, 1, 0).astype('int')

        labl = self.loader.y_test
        labl = (labl / weit)[:, 0]

        score = scoring(date_list=list(self.loader.d_test), weight_list=list(self.loader.w_test), resp_list=list(self.loader.r_test), action_list=list(pred))
        scorl = scoring(date_list=list(self.loader.d_test), weight_list=list(self.loader.w_test), resp_list=list(self.loader.r_test), action_list=list(labl))

        # labl = np.where(np.mean(self.loader.y_test, axis=1) > 0, 1, 0).astype('int')
        # labl = np.where(np.mean(np.where(self.loader.y_test[0] > 0, 1, 0), axis=1)).astype('int')

        acc = sum([1 if p == l else 0 for p, l in zip(pred, labl)]) / float(len(pred))
        fpr, tpr, thresh = roc_curve(labl, pred, pos_label=1)
        auc = area_under_roc(fpr, tpr)
        print(f'score pred={score}, score inf={scorl}')
        print(f'pred={pred.shape}, {pred[:3]}..., label={labl.shape}, {labl[:3]}...  -->  score={score}, resp-acc={acc}, resp-auc={auc}')

    def run_single(self, ck, run_train=False):
        clf = torch.load(f'checkpoints/mlp-spand-{ck}.pkl')
        clf.eval()

        if run_train:
            for param in clf.parameters():
                param.requires_grad = False
            pred = clf(torch.from_numpy(self.loader.X_train[:100000]).float().cuda()).cpu().detach().numpy()

            # pred = np.where(pred[:, 0] > 0.5, 1, 0).astype('int')
            pred = np.where(np.mean(np.where(pred > 0.5, 1, 0), axis=1) > 0.5, 1, 0).astype('int')

            # labl = self.loader.y_test[:, 0]
            labl = self.loader.y_train[:, 0][:100000]
            score = scoring(date_list=list(self.loader.d_train[:100000]), weight_list=list(self.loader.w_train[:100000]),
                            resp_list=list(self.loader.r_train[:100000]), action_list=list(pred))
            scorl = scoring(date_list=list(self.loader.d_train[:100000]), weight_list=list(self.loader.w_train[:100000]),
                            resp_list=list(self.loader.r_train[:100000]), action_list=list(labl))
            acc = sum([1 if p == l else 0 for p, l in zip(pred, labl)]) / float(len(pred))
            fpr, tpr, thresh = roc_curve(labl, pred, pos_label=1)
            auc = area_under_roc(fpr, tpr)
        else:
            for param in clf.parameters():
                param.requires_grad = False
            pred = clf(torch.from_numpy(self.loader.X_test).float().cuda()).cpu().detach().numpy()

            # pred = np.where(pred[:, 0] > 0.5, 1, 0).astype('int')
            pred = np.where(np.mean(np.where(pred > 0.5, 1, 0), axis=1) > 0.5, 1, 0).astype('int')

            # labl = self.loader.y_test[:, 0]
            labl = self.loader.y_test[:, 0]
            score = scoring(date_list=list(self.loader.d_test),
                            weight_list=list(self.loader.w_test),
                            resp_list=list(self.loader.r_test), action_list=list(pred))
            scorl = scoring(date_list=list(self.loader.d_test),
                            weight_list=list(self.loader.w_test),
                            resp_list=list(self.loader.r_test), action_list=list(labl))
            acc = sum([1 if p == l else 0 for p, l in zip(pred, labl)]) / float(len(pred))
            fpr, tpr, thresh = roc_curve(labl, pred, pos_label=1)
            auc = area_under_roc(fpr, tpr)
        return score, scorl, acc, auc

    def run(self, run_train=False):
        res = {}
        for i in range(1, 11):
            try:
                sl, al, ul, ll = [], [], [], []
                ck = i * 100
                for j in range(1):
                    s, l, a, u = self.run_single(ck, run_train=run_train)
                    sl.append(s[0])
                    al.append(a)
                    ul.append(u)
                    ll.append(l[0])
                    # print(f'{i}th, {j} round score={s}, acc={a}, auc={u}')
                res[f'{ck}'] = {
                    'score': sum(sl) / float(len(sl)),
                    'scorl': sum(ll) / float(len(ll)),
                    'acc': sum(al) / float(len(al)),
                    'auc': sum(ul) / float(len(ul)),
                }
                print(f'avg res at {ck} = {res[str(ck)]}')
            except Exception as e:
                print(e)
        with open('output/mlp-1000-result-spand.json', 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    tester = Tester()
    tester.test()



