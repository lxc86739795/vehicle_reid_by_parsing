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


class VehicleOneM(ImageDataset):
    """
    vehicleid
    Reference:
    Haiyun Guo, Chaoyang Zhao, Zhiwei Liu, Jinqiao Wang, Hanqing Lu: Learning coarse-to-fine structured feature embedding for vehicle re-identification. AAAI 2018.
    URL: http://www.nlpr.ia.ac.cn/iva/homepage/jqwang/Vehicle1M.htm

    Dataset statistics:
    # identities: 55527
    # images: 936051
    """
    dataset_dir = 'Vehicle-1M'

    def __init__(self, root='/home/liuxinchen3/notespace/data', verbose=True, **kwargs):
        self.image_dir = osp.join(root, self.dataset_dir, 'image_jpg')
        self.train_list = osp.join(root, self.dataset_dir, 'train-test-split/train_list.txt')
        self.test_list = osp.join(root, self.dataset_dir, 'train-test-split/test_full.txt')

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

        super(VehicleOneM, self).__init__(train, query, gallery, **kwargs)

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
            imgid = line.split(' ')[0].split('/')[1].split('.')[0]
            if relabel: vid = vid2label[vid]
            img_path = osp.join(self.image_dir, line.split(' ')[0].split('.')[0] + '.jpg')
            dataset.append((img_path, int(vid), int(imgid)))

        # print(dataset)
        # assert len(dataset) == len(img_list_lines)
        random.shuffle(dataset)
        vid_container = set()
        if relabel:
            return dataset
        else:
            query = []
            gallery = []
            for sample in dataset:
                if sample[1] not in vid_container:
                    vid_container.add(sample[1])
                    query.append(sample)

            return query, dataset
