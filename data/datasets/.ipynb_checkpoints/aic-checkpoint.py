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


class AICity19(ImageDataset):
    """
    AICYTY
    Reference:
    Zheng et al. CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification. CVPR 2019.
    URL: https://github.com/zhengthomastang

    Dataset statistics:
    # identities: 666
    # images: 36935 (train) + 1052 (query) + 18290 (gallery)
    # in practice the query and gallery is from veri
    """
    dataset_dir = 'aic19'

    def __init__(self, root='/home/liuxinchen3/notespace/data', verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train_offset')
        self.query_dir = osp.join(self.dataset_dir, 'image_query_eval')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test_eval')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        super(AICity19, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #print('img_path:', img_path, ' pid : ', pid, ' camid : ', camid)
            #assert 0 <= pid
            #assert 0 <= camid
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
