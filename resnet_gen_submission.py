# coding:utf-8

from hpacellseg.utils import label_cell
from torchvision.transforms import transforms
from PIL import Image
from collections import defaultdict

import pycocotools.mask as coco_mask
import hpacellseg.cellsegmentator as cellsegmentator
import matplotlib.pyplot as plt
import os
import numpy as np
import base64
import zlib
import torch
import cv2
import pandas as pd
import math


data_root = 'I:/datasets/kaggle/human-protein-atlas/test'
segmentor = cellsegmentator.CellSegmentator('data/nuclei-model.pth', 'data/cell-model.pth', padding=True)
prefix_list_all = [x.split('_')[0] for x in os.listdir(data_root) if x.endswith('red.png')]
clf = torch.load(f'checkpoints/model.resnet50.singlecell.25.pkl').cuda()
clf.eval()


def binary_mask_to_ascii(mask, mask_val=1):
    """Converts a binary mask into OID challenge encoding ascii text."""
    mask = np.where(mask == mask_val, 1, 0).astype(np.bool)

    # check input mask --
    if mask.dtype != np.bool:
        raise ValueError(f"encode_binary_mask expects a binary mask, received dtype == {mask.dtype}")

    mask = np.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(f"encode_binary_mask expects a 2d mask, received shape == {mask.shape}")

    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(np.uint8)
    mask_to_encode = np.asfortranarray(mask_to_encode)

    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str.decode()


def valid_size(shape):
    s1, s2 = shape[0], shape[1]
    sa, sb = max(s1, s2), min(s1, s2)
    return True if (float(sa) / sb) < 1.5 else False


df = pd.read_csv('data/train.csv').fillna('')
label_dict = {row[0]: [int(x) for x in row[1].split('|')] for row in df.values}

save = []

step = math.ceil(len(prefix_list_all) / float(10))
for ep in range(step):
    prefix_list = prefix_list_all[ep*50:(ep+1)*50]
    print(f'{ep}th infering batch...')
    rpath = [f'{data_root}/{x}_red.png' for x in prefix_list]
    gpath = [f'{data_root}/{x}_green.png' for x in prefix_list]
    bpath = [f'{data_root}/{x}_blue.png' for x in prefix_list]
    ypath = [f'{data_root}/{x}_yellow.png' for x in prefix_list]
    imgs = [rpath, ypath, bpath]
    print(f'input size: {len(imgs[0])}, {len(imgs[1])}, {len(imgs[2])}')

    segs = segmentor.pred_cells(imgs)
    seg_nuc = segmentor.pred_nuclei(imgs[2])
    trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i in range(len(seg_nuc)):
        prefix = prefix_list[i]
        r = cv2.imread(f'{data_root}/{prefix}_red.png', cv2.IMREAD_GRAYSCALE)
        g = cv2.imread(f'{data_root}/{prefix}_green.png', cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(f'{data_root}/{prefix}_blue.png', cv2.IMREAD_GRAYSCALE)
        w, h = b.shape
        img = np.transpose(np.array([r.transpose(), g.transpose(), b.transpose()]))
        nuclei_mask, cell_mask = label_cell(seg_nuc[i], segs[i])
        cell_encode = [(binary_mask_to_ascii(cell_mask, mask_val=cell_id), cell_id) for cell_id in range(1, cell_mask.max() + 1)]
        single_masks = [cell_mask == lb for lb in range(1, cell_mask.max() + 1)]
        pred_count = defaultdict(int)
        submission_str = []
        for j in range(len(single_masks)):
            x, y = np.where(single_masks[j])
            index = np.meshgrid(np.arange(min(x), max(x)+1), np.arange(min(y), max(y)+1), indexing='xy')
            cropped = img[index]
            cropped_mask = single_masks[j][index]
            encoded = cell_encode[j][0]
            if valid_size(cropped.shape):
                inputs = trans(Image.fromarray(cropped)).unsqueeze(0).cuda()
                pred = clf(inputs).detach().cpu().numpy()[0]
                pred_idx = [i for i in range(len(pred)) if pred[i] > 0.5]
                if not pred_idx:
                    pred_sort = sorted([(i, x) for i, x in enumerate(pred) if x > 0], key=lambda x: x[1], reverse=True)
                    pred_idx = [pred_sort[0][0]] if pred_sort else []
                for e in pred_idx:
                    pred_count[e] += 1
        pred_curr = [k for k, v in pred_count.items() if float(v)/len(single_masks) > 0.25]
        for p in pred_curr:
            for enc in cell_encode:
                submission_str.extend([str(p), '0.5', enc[0]])
        save.append([prefix, w, h, ' '.join(submission_str)])
        label_idx = label_dict.get(prefix, [])
        print(f'id={prefix}, pred={sorted(pred_curr)}, label={sorted(label_idx)}, save size={len(save)}')

    savedf = pd.DataFrame(save)
    savedf.to_csv('output/submission-0.csv', index=False, header=['ID', 'ImageWidth', 'ImageHeight', 'PredictionString'])


