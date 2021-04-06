# coding:utf-8

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

console = sys.stdout

with open('log.txt', 'a', encoding='utf-8') as logfile:

    dpath = 'I:/datasets/kaggle/human-protein-atlas/train-single-cell/cells'
    cpath = 'data/cell_df.csv'
    batch_size = 32


    class DSet(Dataset):
        def __init__(self):
            dd = pd.read_csv(cpath).fillna('').values
            df = [x for x in dd if self.valid_size(x)]
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
            img = Image.open(os.path.join(dpath, f'{self.pic_ids[index]}_{self.cel_ids[index]}.jpg'))
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
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    dataset = DSet()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(f'data init finished with size: {len(dataset.labels)}')
    batch_num = len(dataset.labels) // batch_size

    for e in range(50):
        for idx, (l, b) in enumerate(dataloader):
            l, b = l.float().cuda(), b.cuda()
            optimizer.zero_grad()
            p = net(b.cuda())
            loss = criterion(p, l)
            loss.backward()
            optimizer.step()
            sys.stdout = console
            print(f'epoch={e}, batch={idx}/{batch_num}, loss={loss.data.item()}')
            sys.stdout = logfile
            print(f'epoch={e}, batch={idx}/{batch_num}, loss={loss.data.item()}')
        torch.save(net, f'checkpoints/model.resnet50.singlecell.{e}.pkl')


