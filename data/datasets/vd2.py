# encoding: utf-8
"""
@author:  Xinchen Liu
@contact: lxc86739795@gmail.com
"""

import glob
import re

import os.path as osp
import random
from .bases import ImageDataset
import warnings


class VD2(ImageDataset):
    """
    vehicleid
    Reference:
    Ke Yan, Yonghong Tian, Yaowei Wang, Wei Zeng, Tiejun Huang: Exploiting Multi-Grain Ranking Constraints for Precisely Searching Visually-similar Vehicles. ICCV 2017.
    URL: https://pkuml.org/resources/pku-vd.html

    Dataset statistics:
    # identities: 77,963
    # images: 690,518
    """
    dataset_dir = 'PKU-VD'

    def __init__(self, root='/home/liuxinchen3/notespace/data', verbose=True, **kwargs):
        self.image_dir = osp.join(root, self.dataset_dir, 'VD2/image')
        self.train_list = osp.join(root, self.dataset_dir, 'VD2/train_test/trainlist.txt')
        self.test_list = osp.join(root, self.dataset_dir, 'VD2/train_test/testlist.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.test_list
        ]

        self.check_before_run(required_files)

        query, gallery = self._process_dir(self.test_list, relabel=False)
        train = self._process_dir(self.train_list, relabel=True)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(VD2, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, img_list, relabel=False):

        vid_container = set()
        img_list_lines = open(img_list, 'r').readlines()
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split(' ')[1]
            vid_container.add(vid)
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for idx, line in enumerate(img_list_lines):
            # if idx < 10:
            line = line.strip()
            vid = line.split(' ')[1]
            imgid = line.split(' ')[0]
            if relabel: vid = vid2label[vid]
            img_path = osp.join(self.image_dir, line.split(' ')[0] + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        # print(dataset)
        # assert len(dataset) == len(img_list_lines)
        random.shuffle(dataset)
        vid_container = set()
        if relabel:
            return dataset
        else:
            dataset = dataset[:100000]
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    query.append(sample)

            return query, dataset
