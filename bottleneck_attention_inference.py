# coding:utf-8

from config import *
from torchvision import transforms
from PIL import Image

import cv2
import pandas as pd
import torch
import os

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
model = torch.load('checkpoints\\model.fast.ai.003.pkl')
model = model.eval().cuda()

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats),
    transforms.CenterCrop((256, 256)),
])

print(model)

img_files = [os.path.join(DATA_PATH, 'train', x) for x in os.listdir(os.path.join(DATA_PATH, 'train')) if x.endswith('green.png')][:5]
# img = cv2.imread(img_files[0], )
# w, h = img.shape[:2]
# img = img[w//2-128: w//2+128, h//2-128: h//2+128]
# img = torch.from_numpy(img).permute(2, 0, 1)
# img = trans(img)
img = Image.open(img_files[0]).convert('RGB')
img = trans(img)
tensor = torch.unsqueeze(img, 0).float().cuda()
print(tensor.shape)
res = model(tensor)
print(res)





