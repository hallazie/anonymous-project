# coding:utf-8

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from criterion import f1_score
from blocks import *
from focal_loss.focal_loss import FocalLoss
from sklearn.model_selection import train_test_split

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
log_path = 'log-with-eval-resnet50.txt'


class DSet(Dataset):
    def __init__(self, id_set, task):
        self.id_set = id_set
        dd = pd.read_csv(cpath).fillna('').values
        df = [x for x in dd if x[0] in self.id_set]
        print(f'valid {task} sample: {len(dd)} --> {len(df)}')
        self.pic_ids = [x[0] for x in df]
        self.labels = [[int(y.strip()) for y in x[1].split('|')] for x in df]
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size * 2, img_size * 2)),
            transforms.CenterCrop((img_size, img_size)),
            transforms.Normalize(mean=[0.456], std=[0.224]),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
        ])

    def __getitem__(self, index):
        label = np.array(self.labels[index])
        img = Image.open(os.path.join(root_path, f'{self.pic_ids[index]}_green.png')).convert('L')
        label_tensor = np.zeros((19, ))
        label_tensor[label] = 1
        img_tensor = self.trans(img)
        return label_tensor, img_tensor

    def __len__(self):
        return len(self.labels)


f = open(log_path, 'w')
f.close()
with open(log_path, 'a', encoding='utf-8') as logfile:

    root_path = 'I:/datasets/kaggle/human-protein-atlas/train'
    cpath = 'data/train.csv'
    batch_size = 4
    img_size = 512
    seed = 30

    total_ids = [x.split('_green')[0] for x in os.listdir(root_path) if x.endswith('green.png')]
    train_ids, test_ids = train_test_split(total_ids, test_size=0.1, random_state=seed)
    random.shuffle(train_ids)
    dev_ids = train_ids[:len(test_ids)]

    tinp = torch.zeros((1, 1, img_size, img_size))
    densenet = models.resnet50(pretrained=True)
    densenet.conv1 = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)   # change input channel to 1
    # densenet = models.densenet121(pretrained=True)
    # densenet.features[0] = nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)   # change input channel to 1

    backbone = nn.Sequential(*list(densenet.children())[:-2])
    tout = backbone(tinp)
    hidden_size = tout.shape[1]

    net = nn.Sequential(
        *list(densenet.children())[:-2],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(hidden_size, 19),
        nn.Sigmoid(),
    )

    net = net.cuda()
    net.train()

    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.2, last_epoch=-1)

    dataset_train = DSet(train_ids, 'train')
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    print(f'train data init finished with size: {len(dataset_train.labels)}')
    batch_num = len(dataset_train.labels) // batch_size

    dataset_dev = DSet(dev_ids, 'dev')
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True)
    print(f'dev data init finished with size: {len(dataset_dev.labels)}')
    batch_num_dev = len(dataset_dev.labels) // batch_size

    dataset_test = DSet(test_ids, 'test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)
    print(f'dev data init finished with size: {len(dataset_test.labels)}')
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
        torch.save(net, f'ckpt/checkpoints-resnet50/model.resnet50.wholecell.{e}.pkl')

        scheduler.step()
        # optimizer.step()

        ll, pp = [], []
        for idx, (l, b) in enumerate(dataloader_dev):
            l, b = l.float().cuda(), b.cuda()
            p = net(b)
            l_idx = l.cpu().detach().numpy()
            p_idx = p.cpu().detach().numpy()
            l_ = [[i for i, v in enumerate(t) if v > 0.5] for t in l_idx]
            p_ = [[i for i, v in enumerate(t) if v > 0.5] for t in p_idx]
            ll.extend(l_)
            pp.extend(p_)
            sys.stdout = console
            print(f'epoch={e} dev inference {idx}/{batch_num_dev} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')
            sys.stdout = logfile
            print(f'epoch={e} dev inference {idx}/{batch_num_dev} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')

        sys.stdout = console
        print(f'epoch={e} dev final f1 score on test set ({dataloader_dev.__len__()}): {f1_score(ll, pp)}')
        sys.stdout = logfile
        print(f'epoch={e} dev final f1 score on test set ({dataloader_dev.__len__()}): {f1_score(ll, pp)}')

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
            print(f'epoch={e} test inference {idx}/{batch_num_test} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')
            sys.stdout = logfile
            print(f'epoch={e} test inference {idx}/{batch_num_test} finished with f1-score: {f1_score(l_, p_)}, p={p_}, l={l_}')

        sys.stdout = console
        print(f'epoch={e} test final f1 score on test set ({dataloader_test.__len__()}): {f1_score(ll, pp)}')
        sys.stdout = logfile
        print(f'epoch={e} test final f1 score on test set ({dataloader_test.__len__()}): {f1_score(ll, pp)}')







