# coding:utf-8

from hpacellseg.utils import label_cell, label_nuclei

import pycocotools.mask as coco_mask
import hpacellseg.cellsegmentator as cellsegmentator
import matplotlib.pyplot as plt
import os
import numpy as np
import base64
import zlib


data_root = 'I:/datasets/kaggle/human-protein-atlas/test'
segmentor = cellsegmentator.CellSegmentator('data/nuclei-model.pth', 'data/cell-model.pth', padding=True)
prefix_list = [x.split('_')[0] for x in os.listdir(data_root) if x.endswith('red.png')][:10]

rpath = [f'{data_root}/{x}_red.png' for x in prefix_list]
gpath = [f'{data_root}/{x}_green.png' for x in prefix_list]
bpath = [f'{data_root}/{x}_blue.png' for x in prefix_list]
ypath = [f'{data_root}/{x}_yellow.png' for x in prefix_list]
imgs = [rpath, ypath, bpath]
print(len(imgs[0]), len(imgs[1]), len(imgs[2]))

segs = segmentor.pred_cells(imgs)
seg_nuc = segmentor.pred_nuclei(imgs[2])


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


for i in range(len(seg_nuc)):
    prefix = prefix_list[i]
    nuclei_mask, cell_mask = label_cell(seg_nuc[i], segs[i])
    cell_encode = [(binary_mask_to_ascii(cell_mask, mask_val=cell_id), cell_id) for cell_id in range(1, cell_mask.max() + 1)]
    single_masks = [cell_mask == lb for lb in range(1, cell_mask.max() + 1)][:4]
    print(prefix)
    plt.subplot(221)
    plt.imshow(single_masks[0])
    plt.subplot(222)
    plt.imshow(single_masks[1])
    plt.subplot(223)
    x, y = np.where(single_masks[0])
    index = np.meshgrid(np.arange(min(x), max(x)+1), np.arange(min(y), max(y)+1), indexing='xy')
    crop0 = single_masks[0][index]
    plt.imshow(crop0)
    plt.subplot(224)
    x, y = np.where(single_masks[0])
    index = np.meshgrid(np.arange(min(x), max(x)+1), np.arange(min(y), max(y)+1), indexing='xy')
    crop1 = single_masks[1][index]
    plt.imshow(crop1)
    plt.title(prefix)
    plt.show()

