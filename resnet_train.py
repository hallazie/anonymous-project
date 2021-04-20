# coding:utf-8

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from criterion import f1_score

import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import random


console = sys.stdout

with open('log-with-eval2.txt', 'a', encoding='utf-8') as logfile:

    train_path = 'I:/datasets/kaggle/human-protein-atlas/train-single-cell/cells'
    test_path = 'I:/datasets/kaggle/human-protein-atlas/train-single-cell/cells-test'
    cpath = 'data/cell_df.csv'
    batch_size = 24


    class DSet(Dataset):
        def __init__(self, dpath):
            self.dpath = dpath
            id_set = set([x.split('.')[0] for x in os.listdir(dpath) if x.endswith('jpg')])
            dd = pd.read_csv(cpath).fillna('').values
            df = [x for x in dd if self.valid_size(x) if f'{x[0]}_{x[4]}' in id_set]
            print(f'valid training sample: {len(dd)} --> {len(df)}')
            self.pic_ids = [x[0] for x in df]
            self.cel_ids = [x[4] for x in df]
            self.labels = [[int(y) for y in x[5].split('|')] for x in df]
            self.trans = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        @staticmethod
        def valid_size(row):
            s1, s2 = row[-2], row[-1]
            sa, sb = max(s1, s2), min(s1, s2)
            return True if (float(sa) / sb) < 1.5 else False

        def __getitem__(self, index):
            label = np.array(self.labels[index])
            img = Image.open(os.path.join(self.dpath, f'{self.pic_ids[index]}_{self.cel_ids[index]}.jpg'))
            label_tensor = np.zeros((19, ))
            label_tensor[label] = 1
            img_tensor = self.trans(img)
            return label_tensor, img_tensor

        def __len__(self):
            return len(self.labels)


    tinp = torch.zeros((1, 3, 256, 256))

    resnet18 = models.resnet50(pretrained=True)
    # resnet18 = models.resnet18(pretrained=True)

    backbone = nn.Sequential(*list(resnet18.children())[:-2])
    tout = backbone(tinp)
    hidden_size = tout.shape[1]

    net = nn.Sequential(
        *list(resnet18.children())[:-2],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(hidden_size, 19),
        nn.Sigmoid()
    )

    # net = torch.load('checkpoints/model.resnet50.singlecell.pkl')
    net = net.cuda()
    net.train()

    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.2, last_epoch=-1)

    dataset_train = DSet(train_path)
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    print(f'data init finished with size: {len(dataset_train.labels)}')
    batch_num = len(dataset_train.labels) // batch_size

    dataset_test = DSet(test_path)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    print(f'data init finished with size: {len(dataset_test.labels)}')
    batch_num_test = len(dataset_test.labels) // batch_size

    for e in range(50):
        for idx, (l, b) in enumerate(dataloader_train):
            l, b = l.float().cuda(), b.cuda()
            optimizer.zero_grad()
            p = net(b.cuda())
            loss = criterion(p, l)
            loss.backward()
            optimizer.step()
            sys.stdout = console
            print(f'epoch={e}, lr={optimizer.param_groups[0].get("lr")}, batch={idx}/{batch_num}, loss={loss.data.item()}')
            sys.stdout = logfile
            print(f'epoch={e}, lr={optimizer.param_groups[0].get("lr")}, batch={idx}/{batch_num}, loss={loss.data.item()}')
        torch.save(net, f'checkpoints/model.resnet50.singlecell.{e}.pkl')

        scheduler.step()

        ll, pp = [], []
        for idx, (l, b) in enumerate(dataloader_test):
            l, b = l.float().cuda(), b.cuda()
            p = net(b)
            l_idx = l.cpu().detach().numpy()
            p_idx = p.cpu().detach().numpy()
            l_ = [[i for i, v in enumerate(t) if v > 0.5] for t in l_idx]
            p_ = [[i for i, v in enumerate(t) if v > 0.5] for t in p_idx]
            ll.extend(l_)
            pp.extend(p_)
            sys.stdout = console
            print(f'epoch={e} inference {idx}/{batch_num_test} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')
            sys.stdout = logfile
            print(f'epoch={e} inference {idx}/{batch_num_test} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')

        sys.stdout = console
        print(f'epoch={e} finial f1 score on test set ({dataloader_test.__len__()}): {f1_score(ll, pp)}')
        sys.stdout = logfile
        print(f'epoch={e} finial f1 score on test set ({dataloader_test.__len__()}): {f1_score(ll, pp)}')










