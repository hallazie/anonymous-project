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


data_root = 'I:/datasets/kaggle/human-protein-atlas/train'
segmentor = cellsegmentator.CellSegmentator('data/nuclei-model.pth', 'data/cell-model.pth', padding=True)
prefix_list_all = [x.split('_')[0] for x in os.listdir(data_root) if x.endswith('red.png')]
clf = torch.load(f'checkpoints-2/model.resnet50.singlecell.5.pkl').cuda()
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


df = pd.read_csv('data/train.csv').fillna('')
label_dict = {row[0]: [int(x) for x in row[1].split('|')] for row in df.values}

if len(prefix_list_all) != 559:
    prefix_list_all = prefix_list_all[:10]
save = []
step_size = 5
step = math.ceil(len(prefix_list_all) / float(step_size))

for ep in range(step):
    prefix_list = prefix_list_all[ep*step_size:(ep+1)*step_size]
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
        transforms.Normalize(mean=[0.456], std=[0.224])
    ])

    for i in range(len(seg_nuc)):
        prefix = prefix_list[i]
        g = cv2.imread(f'{data_root}/{prefix}_green.png', cv2.IMREAD_GRAYSCALE)
        w, h = g.shape
        img = np.transpose(g.transpose())
        nuclei_mask, cell_mask = label_cell(seg_nuc[i], segs[i])
        cell_encode = [(binary_mask_to_ascii(cell_mask, mask_val=cell_id), cell_id) for cell_id in range(1, cell_mask.max() + 1)]
        single_masks = [cell_mask == lb for lb in range(1, cell_mask.max() + 1)]
        pred_count = defaultdict(int)
        submission_str = []
        pred_curr = set()
        for j in range(len(single_masks)):
            x, y = np.where(single_masks[j])
            index = np.meshgrid(np.arange(min(x), max(x)+1), np.arange(min(y), max(y)+1), indexing='xy')
            cropped = img[index]
            cropped_mask = single_masks[j][index]
            encoded = cell_encode[j][0]
            inputs = trans(Image.fromarray(cropped)).unsqueeze(0).cuda()
            pred = clf(inputs).detach().cpu().numpy()[0]
            pred_res = [(i, pred[i]) for i in range(len(pred)) if pred[i] > 0.1]
            submission_str.extend([f'{str(x[0])} {str(x[1])} {encoded}' for x in pred_res])
            pred_curr |= set([x for x in pred_res])
        save.append([prefix, w, h, ' '.join(submission_str)])
        label_idx = label_dict.get(prefix, [])
        print(f'id={prefix}, pred={sorted(pred_curr)}, label={sorted(label_idx)}, save size={len(save)}')

savedf = pd.DataFrame(save)
# savedf.to_csv('/kaggle/working/submission.csv', index=False, header=['ID', 'ImageWidth', 'ImageHeight', 'PredictionString'])
savedf.to_csv('output/submission-1.csv', index=False, header=['ID', 'ImageWidth', 'ImageHeight', 'PredictionString'])
print('submission has saved...')
