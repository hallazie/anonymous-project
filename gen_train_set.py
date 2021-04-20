# coding:utf-8

from hpacellseg.utils import label_cell
from PIL import Image
from collections import defaultdict

import pycocotools.mask as coco_mask
import hpacellseg.cellsegmentator as cellsegmentator
import matplotlib.pyplot as plt
import os
import numpy as np
import base64
import zlib
import cv2
import pandas as pd
import math


def valid_size(shape):
    s1, s2 = shape[0], shape[1]
    sa, sb = max(s1, s2), min(s1, s2)
    return True if (float(sa) / sb) < 1.5 else False


df = pd.read_csv('data/train.csv').fillna('')
id_list = [x for x in df.values if '|' not in x[1]]
data_root = 'I:/datasets/kaggle/human-protein-atlas/train'
save_root = 'I:/datasets/kaggle/human-protein-atlas/train-single-cell/cell-single-label'
segmentor = cellsegmentator.CellSegmentator('data/nuclei-model.pth', 'data/cell-model.pth', padding=True)

save = []
step_size = 10
step = math.ceil(len(id_list) / float(step_size))
for ep in range(step):
    try:
        prefix_list = [x[0] for x in id_list[ep * step_size:(ep + 1) * step_size]]
        label_list = [x[1] for x in id_list[ep * step_size:(ep + 1) * step_size]]
        print(f'{ep}th infering batch...')
        rpath = [f'{data_root}/{x}_red.png' for x in prefix_list]
        gpath = [f'{data_root}/{x}_green.png' for x in prefix_list]
        bpath = [f'{data_root}/{x}_blue.png' for x in prefix_list]
        ypath = [f'{data_root}/{x}_yellow.png' for x in prefix_list]
        imgs = [rpath, ypath, bpath]
        print(f'input size: {len(imgs[0])}, {len(imgs[1])}, {len(imgs[2])}')

        segs = segmentor.pred_cells(imgs)
        seg_nuc = segmentor.pred_nuclei(imgs[2])

        for i in range(len(seg_nuc)):
            prefix = prefix_list[i]
            label = label_list[i]
            g = cv2.imread(f'{data_root}/{prefix}_green.png', cv2.IMREAD_GRAYSCALE)
            w, h = g.shape
            nuclei_mask, cell_mask = label_cell(seg_nuc[i], segs[i])
            single_masks = [cell_mask == lb for lb in range(1, cell_mask.max() + 1)]
            for j in range(len(single_masks)):
                x, y = np.where(single_masks[j])
                index = np.meshgrid(np.arange(min(x), max(x) + 1), np.arange(min(y), max(y) + 1), indexing='xy')
                cropped = g[index]
                ws, hs = cropped.shape
                if not valid_size(cropped.shape):
                    continue
                line = [prefix, 0, 0, 0, j, label, ws, hs]
                save.append(line)
                print(len(save), line)
                cv2.imwrite(os.path.join(save_root, f'{prefix}_{j}.png'), cropped)
    except Exception as e:
        print(e)

dd = pd.DataFrame(save)
dd.to_csv('data/cell_df_single.csv', index=False, header=['image_id', 'r_mean', 'g_mean', 'b_mean', 'cell_id', 'image_labels', 'size1', 'size2'])
