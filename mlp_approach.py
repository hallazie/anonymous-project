# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: mlp_approach.py
# @time: 2021/1/24 14:39
# @desc:

from dataloader import DataLoader
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


class MLPModel(nn.Module):
    def __init__(self, input_size, neural_size):
        super(MLPModel, self).__init__()
        neural_size_fill = [input_size] + neural_size
        neural_list = [
            (nn.Linear(in_features=neural_size_fill[i], out_features=neural_size_fill[i + 1], bias=True),
             nn.BatchNorm1d(num_features=neural_size_fill[i + 1]),
             nn.Dropout(0.125),
             nn.Tanh())
            for i in range(len(neural_size_fill) - 1)]
        neural_list = [x for y in neural_list for x in y]
        self.bone = nn.Sequential(*neural_list)
        self.out = nn.Sequential(nn.Linear(in_features=neural_size[-1], out_features=1), nn.Sigmoid())

    def forward(self, x):
        y = self.bone(x)
        y = self.out(y)
        return y


def run():
    loader = DataLoader()
    epoch = 5000
    batch_size = 4096
    hidden_units = [164, 164, 256, 256, 256, 128, 32]
    drop_rates = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]
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








