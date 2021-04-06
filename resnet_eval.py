# coding:utf-8

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from criterion import f1_score

import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
import os
import sys

console = sys.stdout

with open('log.txt', 'a', encoding='utf-8') as logfile:

    dpath = 'I:/datasets/kaggle/human-protein-atlas/train-single-cell/cells'
    cpath = 'data/cell_df.csv'
    batch_size = 24


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


    net = torch.load('checkpoints/model.resnet50.singlecell.29.pkl')
    # net.requires_grad = False
    net.eval()

    dataset = DSet()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(f'data init finished with size: {len(dataset.labels)}')
    batch_num = len(dataset.labels) // batch_size

    ll, pp = [], []

    for idx, (l, b) in enumerate(dataloader):
        l, b = l.float().cuda(), b.cuda()
        p = net(b.cuda())
        l_idx = l.cpu().detach().numpy()
        p_idx = p.cpu().detach().numpy()
        l_ = [[i for i, v in enumerate(t) if v > 0.5] for t in l_idx]
        p_ = [[i for i, v in enumerate(t) if v > 0.5] for t in p_idx]
        ll.extend(l_)
        pp.extend(p_)
        print(f'inference {idx}/{batch_num} finished with f1-score: {f1_score(l_, p_)}')

    print(f'finial f1 score on train set: {f1_score(ll, pp)}')

