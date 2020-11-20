# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *


# def build_transforms(cfg, is_train=True):
#     res = []
#     if is_train:
#         res.append(T.Resize(cfg.INPUT.SIZE_TRAIN))
#         if cfg.INPUT.DO_FLIP:
#             res.append(T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB))
#         if cfg.INPUT.DO_PAD:
#             res.extend([T.Pad(cfg.INPUT.PADDING, padding_mode=cfg.INPUT.PADDING_MODE), 
#                         T.RandomCrop(cfg.INPUT.SIZE_TRAIN)])
#         if cfg.INPUT.DO_LIGHTING:
#             res.append(T.ColorJitter(cfg.INPUT.MAX_LIGHTING, cfg.INPUT.MAX_LIGHTING))
#         # res.append(T.ToTensor())  # to slow
#         if cfg.INPUT.DO_RE:
#             res.append(RandomErasing(probability=cfg.INPUT.RE_PROB))
#     else:
#         res.append(T.Resize(cfg.INPUT.SIZE_TEST))
#         # res.append(T.ToTensor())
#     return T.Compose(res)

def build_transforms(cfg, is_train=True):
    #normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation = 3),
            T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.ToTensor(),
            #normalize_transform,
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
        mask_transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation = 0),
            T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #T.ToTensor()
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mean=[0])
        ])

    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            #T.ToTensor(),
            #normalize_transform
        ])
        mask_transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            #T.ToTensor(),
            #normalize_transform
        ])

    return (transform, mask_transform)