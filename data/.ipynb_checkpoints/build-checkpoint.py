# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import glob
import os
import re

# Changed by Xinchen Liu

from torch.utils.data import DataLoader
from .collate_batch import fast_collate_fn, fast_instance_collate_fn, fast_collate_fn_mask
from .datasets import ImageDataset, InstanceDataset, ImageMaskDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_mask
from .transforms import build_transforms
from .datasets import init_dataset


def get_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    print('prepare training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        # dataset = init_dataset(d, combineall=True)
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)

    tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    data_sampler = None
    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler(tng_set.img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                                num_workers=num_workers, sampler=data_sampler,
                                collate_fn=fast_collate_fn, pin_memory=True)

    print('prepare test set ...')

    val_dataloader_collection, query_names_len_collection, _ = get_test_dataloader(cfg)
    return tng_dataloader, val_dataloader_collection, tng_set.c, query_names_len_collection

def get_test_dataloader(cfg):
    val_tfms = build_transforms(cfg, is_train=False)
    test_dataloader_collection, query_names_len_collection, test_names_collection = list(), list(), list()

    for d in cfg.DATASETS.TEST_NAMES:
        dataset = init_dataset(d)
        query_names, gallery_names = dataset.query, dataset.gallery

        num_workers = cfg.DATALOADER.NUM_WORKERS

        test_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
        test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                                    collate_fn=fast_collate_fn, pin_memory=True)
        test_dataloader_collection.append(test_dataloader)
        query_names_len_collection.append(len(query_names))
        test_names_collection.append(query_names+gallery_names)
    return test_dataloader_collection, query_names_len_collection, test_names_collection

def get_ins_dataloader(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)

    print('prepare instance training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)

    print('prepare test set ...')
    dataset = init_dataset(cfg.DATASETS.TEST_NAMES)
    query_names, gallery_names = dataset.query, dataset.gallery

    tng_set = InstanceDataset(train_img_items, tng_tfms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_sampler = None
    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                                num_workers=num_workers, sampler=data_sampler,
                                collate_fn=fast_instance_collate_fn, pin_memory=True)

    val_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
    val_dataloader = DataLoader(val_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers, 
                                collate_fn=fast_collate_fn, pin_memory=True)
    return tng_dataloader, val_dataloader, tng_set.c, len(query_names)

def get_dataloader_mask(cfg):
    tng_tfms = build_transforms(cfg, is_train=True)
    val_tfms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    print('prepare training set ...')
    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)

    tng_set = ImageMaskDataset(train_img_items, tng_tfms, relabel=True)

    data_sampler = None
    if cfg.DATALOADER.SAMPLER == 'triplet':
        data_sampler = RandomIdentitySampler_mask(tng_set.img_items, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)

    tng_dataloader = DataLoader(tng_set, cfg.SOLVER.IMS_PER_BATCH, shuffle=(data_sampler is None),
                                num_workers=num_workers, sampler=data_sampler,
                                collate_fn=fast_collate_fn_mask, pin_memory=True)

    print('prepare test set ...')

    val_dataloader_collection, query_names_len_collection, _ = get_test_dataloader_mask(cfg)
    return tng_dataloader, val_dataloader_collection, tng_set.c, query_names_len_collection

def get_test_dataloader_mask(cfg):
    val_tfms = build_transforms(cfg, is_train=False)
    test_dataloader_collection, query_names_len_collection, test_names_collection = list(), list(), list()

    for d in cfg.DATASETS.TEST_NAMES:
        dataset = init_dataset(d)
        query_names, gallery_names = dataset.query, dataset.gallery

        num_workers = cfg.DATALOADER.NUM_WORKERS

        test_set = ImageMaskDataset(query_names+gallery_names, val_tfms, relabel=False)
        test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers,
                                    collate_fn=fast_collate_fn_mask, pin_memory=True)
        test_dataloader_collection.append(test_dataloader)
        query_names_len_collection.append(len(query_names))
        test_names_collection.append(query_names+gallery_names)
    return test_dataloader_collection, query_names_len_collection, test_names_collection
