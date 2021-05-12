# coding:utf-8

import pandas as pd
import matplotlib.pyplot as plt

# Python library to interact with the file system.
import os

# Python library for image augmentation
import albumentations as A

# fastai library for computer vision tasks
from fastai.vision.all import *

# Developing and training neural network based deep learning models.
import torch
from torch import nn
from torchvision.models import resnet101

# BotNet
from bottleneck_transformer_pytorch import BottleStack

# Define path to dataset, whose benefit is that this sample is more balanced than original train data.
path = Path('I:\\datasets\\kaggle\\human-protein-atlas\\train')

df = pd.read_csv('I:\\datasets\\kaggle\\human-protein-atlas\\train.csv')
print(df.head())

img_size = 256

# extract the the total number of target labels
labels = [str(i) for i in range(19)]
for x in labels: df[x] = df['Label'].apply(lambda r: int(x in r.split('|')))

# Here a sample of the dataset has been taken, change frac to 1 to train the entire dataset!
dfs = df.sample(frac=1, random_state=42)
dfs = dfs.reset_index(drop=True)
print(len(dfs))


def get_x(r):
    return path / (r['ID'] + '_green.png')


# obtain the targets.
def get_y(r):
    return r['Label'].split('|')


class AlbumentationsTransform(RandTransform):
    '''split_idx is None, which allows for us to say when we're setting our split_idx.
       We set an order to 2 which means any resize operations are done first before our new transform. '''
    split_idx, order = None, 2

    def __init__(self, train_aug, valid_aug, **kwargs):
        super().__init__(**kwargs)
        store_attr()

    # Inherit from RandTransform, allows for us to set that split_idx in our before_call.
    def before_call(self, b, split_idx):
        self.idx = split_idx

    # If split_idx is 0, run the trainining augmentation, otherwise run the validation augmentation.
    def encodes(self, img: PILImage):
        if self.idx == 0:
            aug_img = self.train_aug(image=np.array(img))['image']
        else:
            aug_img = self.valid_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(size):
    return A.Compose([
        # allows to combine RandomCrop and RandomScale
        A.RandomResizedCrop(size, size),

        # Transpose the input by swapping rows and columns.
        A.Transpose(p=0.5),

        # Flip the input horizontally around the y-axis.
        A.HorizontalFlip(p=0.5),

        # Flip the input horizontally around the x-axis.
        A.VerticalFlip(p=0.5),

        # Randomly apply affine transforms: translate, scale and rotate the input.
        A.ShiftScaleRotate(p=0.5),

        # Randomly change hue, saturation and value of the input image.
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),

        # Randomly change brightness and contrast of the input image.
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),

        # CoarseDropout of the rectangular regions in the image.
        A.CoarseDropout(p=0.5),

        # CoarseDropout of the square regions in the image.
        A.Cutout(p=0.5)])


def get_valid_aug(size):
    return A.Compose([
        # Crop the central part of the input.
        A.CenterCrop(size, size, p=1.),

        # Resize the input to the given height and width.
        A.Resize(size, size)], p=1.)


'''The first step item_tfms resizes all the images to the same size (this happens on the CPU) 
   and then batch_tfms happens on the GPU for the entire batch of images. '''
# Transforms we need to do for each image in the dataset
item_tfms = [Resize(img_size), AlbumentationsTransform(get_train_aug(img_size), get_valid_aug(img_size))]

# Transforms that can take place on a batch of images
batch_tfms = [Normalize.from_stats(*imagenet_stats)]

bs = 2

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock(vocab=labels)),  # multi-label target
                   splitter=RandomSplitter(seed=42),  # split data into training and validation subsets.
                   get_x=get_x,  # obtain the input images.
                   get_y=get_y,  # obtain the targets.
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms
                   )

dls = dblock.dataloaders(dfs, bs=bs, num_workers=0)

# # We can call show_batch() to see what a sample of a batch looks like.
# dls.show_batch()

layer = BottleStack(
    # channels in
    dim=256,

    # feature map size
    fmap_size=64,

    # channels out
    dim_out=2048,

    # projection factor
    proj_factor=4,

    # downsample on first layer or not
    downsample=True,

    # number of heads
    heads=4,

    # dimension per head, defaults to 128
    dim_head=128,

    # use relative positional embedding - uses absolute if False
    rel_pos_emb=False,

    # activation throughout the network
    activation=nn.ReLU()
)

# define the backbone architecture
resnet = resnet101()

# extract the backbone layers
backbone = list(resnet.children())

# define the model architecture for BotNet
model = nn.Sequential(*backbone[:5],
                      layer,
                      nn.AdaptiveAvgPool2d((1, 1)),
                      nn.Flatten(1),
                      nn.Linear(2048, 19))

# Group together some dls, a model, and metrics to handle training
learn = Learner(dls, model, metrics=accuracy_multi)

# We can use the fine_tune function to train a model with this given learning rate
learn.fine_tune(10, 0.0008317637839354575)

# Plot training and validation losses.
learn.recorder.plot_loss()

torch.save(model, 'ckpt/checkpoints\\model.fast.ai.003.pkl')
