# --*-- coding:utf-8 --*--
# @author: Xiao Shanghua
# @contact: hallazie@outlook.com
# @file: masking.py
# @time: 2020/7/26 17:27
# @desc:

from scipy import ndimage
from third.lungmask.resunet import UNet
from third.lungmask import lungmask_utils
from config import logger

import skimage
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


class LungMask:
    def __init__(self):
        self.model_path = '../../checkpoints/unet_ltrclobes-3a07043d.pth'
        self._init_env()
        self._init_model()

    def _init_env(self):
        if torch.cuda.is_available():
            logger.info('using GPU for lung mask model')
            self.device = torch.device('cuda')
        else:
            logger.warning("No GPU support available, will use CPU. Note, that this is significantly slower!")
            batch_size = 1
            self.device = torch.device('cpu')

    def _init_model(self):
        state_dict = torch.hub.load_state_dict_from_url(self.model_path, progress=True, map_location=torch.device('cpu'))
        self.model = UNet(n_classes=6, padding=True, depth=5, up_mode='upsample', batch_norm=True, residual=False)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def bbox_3d(labelmap, margin=2):
        shape = labelmap.shape
        r = np.any(labelmap, axis=(1, 2))
        c = np.any(labelmap, axis=(0, 2))
        z = np.any(labelmap, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        rmin -= margin if rmin >= margin else rmin
        rmax += margin if rmax <= shape[0] - margin else rmax
        cmin, cmax = np.where(c)[0][[0, -1]]
        cmin -= margin if cmin >= margin else cmin
        cmax += margin if cmax <= shape[1] - margin else cmax
        zmin, zmax = np.where(z)[0][[0, -1]]
        zmin -= margin if zmin >= margin else zmin
        zmax += margin if zmax <= shape[2] - margin else zmax
        if rmax - rmin == 0:
            rmax = rmin + 1
        return np.asarray([rmin, rmax, cmin, cmax, zmin, zmax])

    @staticmethod
    def simple_bodymask(image):
        mask_threshold = -500
        original_shape = image.shape
        image = ndimage.zoom(image, 128 / np.asarray(image.shape), order=0)
        bodymask = image > mask_threshold
        bodymask = ndimage.binary_closing(bodymask)
        bodymask = ndimage.binary_fill_holes(bodymask, structure=np.ones((3, 3))).astype(int)
        bodymask = ndimage.binary_erosion(bodymask, iterations=2)
        bodymask = skimage.measure.label(bodymask.astype(int), connectivity=1)
        regions = skimage.measure.regionprops(bodymask.astype(int))
        if len(regions) > 0:
            max_region = np.argmax(list(map(lambda x: x.area, regions))) + 1
            bodymask = bodymask == max_region
            bodymask = ndimage.binary_dilation(bodymask, iterations=2)
        real_scaling = np.asarray(original_shape) / 128
        return ndimage.zoom(bodymask, real_scaling, order=0)

    @staticmethod
    def keep_largest_connected_component(mask):
        mask = skimage.measure.label(mask)
        regions = skimage.measure.regionprops(mask)
        resizes = np.asarray([x.area for x in regions])
        max_region = np.argsort(resizes)[-1] + 1
        mask = mask == max_region
        return mask

    @staticmethod
    def reshape_mask(mask, tbox, origsize):
        print(mask.shape)
        exit()
        res = np.ones(origsize) * 0
        resize = [tbox[2] - tbox[0], tbox[3] - tbox[1]]
        imgres = ndimage.zoom(mask, resize / np.asarray(mask.shape), order=0)
        res[tbox[0]:tbox[2], tbox[1]:tbox[3]] = imgres
        return res

    def crop_and_resize(self, img, width=192, height=192):
        b_mask = self.simple_bodymask(img)
        reg = skimage.measure.regionprops(skimage.measure.label(b_mask))
        if len(reg) > 0:
            bbox = np.asarray(reg[0].bbox)
        else:
            bbox = (0, 0, b_mask.shape[0], b_mask.shape[1])
        img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        img = ndimage.zoom(img, np.asarray([width, height]) / np.asarray(img.shape), order=1)
        return img, bbox

    def preprocess(self, image_, resolution=(192, 192)):
        image = np.copy(image_)
        image[image < -1024] = -1024
        image[image > 600] = 600
        image, box = self.crop_and_resize(image, width=resolution[0], height=resolution[1])
        return image, box

    def postprocess(self, image):
        regionmask = skimage.measure.label(image)
        origlabels = np.unique(image)
        origlabels_maxsub = np.zeros((max(origlabels) + 1,), dtype=np.uint32)
        regions = skimage.measure.regionprops(regionmask, image)
        regions.sort(key=lambda x: x.area)
        regionlabels = [x.label for x in regions]
        region_to_lobemap = np.zeros((len(regionlabels) + 1,), dtype=np.uint8)
        for r in regions:
            if r.area > origlabels_maxsub[r.max_intensity]:
                origlabels_maxsub[r.max_intensity] = r.area
                region_to_lobemap[r.label] = r.max_intensity
        for r in regions:
            if (r.area < origlabels_maxsub[r.max_intensity]) and r.area > 2:
                bb = self.bbox_3d(regionmask == r.label)
                sub = regionmask[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                dil = ndimage.binary_dilation(sub == r.label)
                neighbours, counts = np.unique(sub[dil], return_counts=True)
                mapto = r.label
                maxmap = 0
                myarea = 0
                for ix, n in enumerate(neighbours):
                    if n != 0 and n != r.label and counts[ix] > maxmap:
                        maxmap = counts[ix]
                        mapto = n
                        myarea = r.area
                regionmask[regionmask == r.label] = mapto
                if regions[regionlabels.index(mapto)].area == origlabels_maxsub[
                    regions[regionlabels.index(mapto)].max_intensity]:
                    origlabels_maxsub[regions[regionlabels.index(mapto)].max_intensity] += myarea
                regions[regionlabels.index(mapto)].__dict__['_cache']['area'] += myarea
        outmask_mapped = region_to_lobemap[regionmask]
        outmask = np.zeros(outmask_mapped.shape, dtype=np.uint8)
        for i in np.unique(outmask_mapped)[1:]:
            outmask[self.keep_largest_connected_component(outmask_mapped == i)] = i
        return outmask

    def apply(self, image: np.ndarray):
        if type(image) is not np.ndarray:
            raise TypeError('apply takes np.ndarray input, not %s' % str(type(image)))
        tvolslices, xnew_box = self.preprocess(image, resolution=[256, 256])
        tvolslices[tvolslices > 600] = 600
        tvolslices = np.divide((tvolslices + 1024), 1624)
        prediction = np.empty((np.append(0, tvolslices.shape)), dtype=np.uint8)
        with torch.no_grad():
            x = torch.from_numpy(tvolslices).unsqueeze(0).unsqueeze(1).to(self.device)
            x = x.float().to(self.device)
            output = self.model(x)
            pls = torch.max(output, 1)[1].detach().cpu().numpy().astype(np.uint8)
            prediction = np.vstack((prediction, pls))
        outmask = self.postprocess(prediction)
        outmask = np.asarray([self.reshape_mask(outmask[i], xnew_box[i], image.shape) for i in range(outmask.shape[0])], dtype=np.uint8)
        return outmask.astype(np.uint8)[0]


if __name__ == '__main__':
    # import pydicom
    # import matplotlib.pyplot as plt
    # path = 'G:\\datasets\\kaggle\\osic-pulmonary-fibrosis-progression\\train\\ID00130637202220059448013\\19.dcm'
    # dset = sitk.ReadImage(path)
    # data = sitk.GetArrayFromImage(dset)[0]
    # # dset = pydicom.filereader.dcmread(path)
    # # data = dset.pixel_array
    # lung_mask = LungMask()
    # mask = lung_mask.apply(data)
    # plt.subplot(121)
    # plt.imshow(data)
    # plt.subplot(122)
    # plt.imshow(mask)
    # plt.show()
    pass