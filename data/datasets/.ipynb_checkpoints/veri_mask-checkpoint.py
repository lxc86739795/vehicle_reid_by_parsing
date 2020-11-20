# encoding: utf-8
"""
@author:  Xinchen Liu
@contact: lxc86739795@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import ImageMaskDataset
import warnings

# Changed by Xinchen Liu

class VeRi_Mask(ImageMaskDataset):
    """
    VeRi
    Reference:
    Liu et al. A Deep Learning based Approach for Progressive Vehicle Re-Identification. ECCV 2016.
    URL: https://vehiclereid.github.io/VeRi/

    Dataset statistics:
    # identities: 775
    # images: 37746 (train) + 1678 (query) + 11579 (gallery)
    """
    dataset_dir = 'veri'

    def __init__(self, root='/home/liuxinchen3/notespace/data', verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.train_mask_dir = osp.join(self.dataset_dir, 'image_train_mask')
        self.query_mask_dir = osp.join(self.dataset_dir, 'image_query_mask')
        self.gallery_mask_dir = osp.join(self.dataset_dir, 'image_test_mask')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.train_mask_dir,
            self.query_mask_dir,
            self.gallery_mask_dir
        ]

        self.check_before_run(required_files)

        train = self._process_dir(self.train_dir, self.train_mask_dir, relabel=True)
        query = self._process_dir(self.query_dir, self.query_mask_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.gallery_mask_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        
        super(VeRi_Mask, self).__init__(train, query, gallery, **kwargs)

    def _process_dir(self, dir_path, mask_dir, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        mask_paths = glob.glob(osp.join(mask_dir, '*.png'))
        img_paths.sort()
        mask_paths.sort()
#         img_names = [name.strip().split('/')[-1].split('.')[0] for name in img_paths]
#         mask_names = [name.strip().split('/')[-1].split('.')[0] for name in mask_paths]
#         diff = list(set(img_names).difference(set(mask_names)))
#         diff.sort()
#         print(len(diff))
#         print(diff)
        assert len(img_paths) == len(mask_paths), f'len(img_paths) = {len(img_paths)}, len(mask_paths) = {len(mask_paths)}'
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        
        for img_path, mask_path in zip(img_paths, mask_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            
            if pid == -1: continue  # junk images are just ignored
            #print('pid : ', pid, ' camid : ', camid)
            assert 1 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, mask_path, pid, camid))

        return dataset
