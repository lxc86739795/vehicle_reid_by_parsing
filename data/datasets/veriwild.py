# encoding: utf-8
"""
@author:  Xinchen Liu
@contact: lxc86739795@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import ImageDataset
import warnings


class VeRiWild(ImageDataset):
    """
    VeRi-Wild
    Reference:
    Lou et al. A Large-Scale Dataset for Vehicle Re-Identification in the Wild. CVPR 2019.
    URL: https://github.com/PKU-IMRE/VERI-Wild

    Dataset statistics:
    # identities: 40,671
    # images: 416,314
    """
    dataset_dir = 'VERI-Wild'

    def __init__(self, root='/home/liuxinchen3/notespace/data', verbose=True, **kwargs):
        self.image_dir = osp.join(root, self.dataset_dir, 'images')
        self.train_list = osp.join(root, self.dataset_dir, 'train_test_split/train_list.txt')
        self.query_list = osp.join(root, self.dataset_dir, 'train_test_split/test_3000_query.txt')
        self.gallery_list = osp.join(root, self.dataset_dir, 'train_test_split/test_3000.txt')
#         self.query_list = osp.join(root, self.dataset_dir, 'train_test_split/test_5000_query.txt')
#         self.gallery_list = osp.join(root, self.dataset_dir, 'train_test_split/test_5000.txt')
#         self.query_list = osp.join(root, self.dataset_dir, 'train_test_split/test_10000_query.txt')
#         self.gallery_list = osp.join(root, self.dataset_dir, 'train_test_split/test_10000.txt')
        self.vehicle_info = osp.join(root, self.dataset_dir, 'train_test_split/vehicle_info.txt')

        required_files = [
            self.image_dir,
            self.train_list,
            self.query_list,
            self.gallery_list,
            self.vehicle_info
        ]

        self.check_before_run(required_files)

        self.imgid2vid, self.imgid2camid, self.imgid2imgpath = self._process_vehicle(self.vehicle_info)

        query = self._process_dir(self.query_list, relabel=False)
        gallery = self._process_dir(self.gallery_list, relabel=False)
        train = self._process_dir(self.train_list, relabel=True)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(VeRiWild, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, img_list, relabel=False):

        vid_container = set()
        img_list_lines = open(img_list, 'r').readlines()
        for idx, line in enumerate(img_list_lines):
            line = line.strip()
            vid = line.split('/')[0]
            vid_container.add(vid)
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        for idx, line in enumerate(img_list_lines):
            # if idx < 10:
            line = line.strip()
            vid = line.split('/')[0]
            imgid = line.split('/')[1]
            if relabel: vid = vid2label[vid]
            dataset.append((self.imgid2imgpath[imgid], int(vid), int(self.imgid2camid[imgid])))

        # print(dataset)
        # random.shuffle(dataset)
        assert len(dataset) == len(img_list_lines)
#         if relabel == True:
#             return dataset[:len(dataset)//4]
        return dataset

    def _process_vehicle(self, vehicle_info):
        imgid2vid = {}
        imgid2camid = {}
        imgid2imgpath = {}
        vehicle_info_lines = open(vehicle_info, 'r').readlines()

        for idx, line in enumerate(vehicle_info_lines[1:]):
            # if idx < 10:
            vid = line.strip().split('/')[0]
            imgid = line.strip().split(';')[0].split('/')[1]
            camid = line.strip().split(';')[1]
            img_path = osp.join(self.image_dir, vid, imgid + '.jpg')
            imgid2vid[imgid] = vid
            imgid2camid[imgid] = camid
            imgid2imgpath[imgid] = img_path
            # print(idx, vid, imgid, camid, img_path)

        assert len(imgid2vid) == len(vehicle_info_lines)-1
        return imgid2vid, imgid2camid, imgid2imgpath
