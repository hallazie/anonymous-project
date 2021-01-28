# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: mlp_approach.py
# @time: 2021/1/24 14:39
# @desc:

from dataloader_fix import DataLoader
from torch.optim import Adam

import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, inp_size, out_size, hidden_units, drop_rates):
        super(MLP, self).__init__()
        assert len(hidden_units) == len(drop_rates)
        linears = [
            (
                nn.Linear(in_features=inp_size if i == 0 else hidden_units[i-1], out_features=hidden_units[i]),
                nn.BatchNorm1d(num_features=hidden_units[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=drop_rates[i])
            )
            for i in range(len(hidden_units))
        ]
        linears = [x for y in linears for x in y]
        linears.append(nn.Linear(in_features=hidden_units[-1], out_features=out_size))
        linears.append(nn.Sigmoid())
        self.bone = nn.Sequential(*linears)

    def forward(self, x):
        return self.bone(x)


def run():
    loader = DataLoader()
    epoch = 1000
    batch_size = 4096
    hidden_units = [164, 164, 256, 128]
    drop_rates = [0.2, 0.2, 0.2, 0.2]
    lr = 1e-3
    datasize = loader.X_train.shape[0]

    clf = MLP(inp_size=130, out_size=5, hidden_units=hidden_units, drop_rates=drop_rates).cuda()
    # clf = torch.load('checkpoints/mlp-base.pkl').cuda()
    clf.train()
    criterion = nn.BCELoss()
    optimizer = Adam(params=clf.parameters(), lr=lr)

    for e in range(epoch):
        for b in range(datasize // batch_size):
            clf.zero_grad()
            x, y = loader.X_train[b * batch_size:(b + 1) * batch_size], loader.y_train[b * batch_size:(b + 1) * batch_size]
            x = torch.tensor(x).float().cuda()
            # y = torch.unsqueeze(torch.tensor(y), 1).float().cuda()
            y = torch.tensor(y).float().cuda()
            p = clf(x)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            if b % 100 == 0:
                print(f'epoch={e}, batch={b}, loss={loss.data}')
        if (e+1) % 200 == 0:
            torch.save(clf, f'checkpoints/mlp-{e+1}.pkl')


if __name__ == '__main__':
    run()








