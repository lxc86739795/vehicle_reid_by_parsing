# encoding: utf-8

import argparse
import os
import sys
import numpy as np
import random

import torch
from torch.backends import cudnn
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append('.')
from config import cfg
from data.collate_batch import fast_collate_fn, fast_instance_collate_fn
from data.datasets import ImageDataset, InstanceDataset
from data.transforms import build_transforms
from data.datasets import init_dataset
from data.prefetcher import data_prefetcher

from modeling import build_model
from utils.logger import setup_logger


def get_train_dataloader(cfg):
    print('prepare training set ...')
    tng_tfms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_img_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = init_dataset(d)
        train_img_items.extend(dataset.train)

    tng_set = ImageDataset(train_img_items, tng_tfms, relabel=True)

    tng_dataloader = DataLoader(tng_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers, collate_fn=fast_collate_fn, pin_memory=True)

    return tng_dataloader, tng_set.c

def get_test_dataloader(cfg):
    print('prepare test set ...')
    val_tfms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    test_dataloader_collection, query_names_len_collection, test_names_collection = list(), list(), list()
    for d in cfg.DATASETS.TEST_NAMES:
        dataset = init_dataset(d)
        query_names, gallery_names = dataset.query, dataset.gallery

        test_set = ImageDataset(query_names+gallery_names, val_tfms, relabel=False)
        
        test_dataloader = DataLoader(test_set, cfg.TEST.IMS_PER_BATCH, num_workers=num_workers, collate_fn=fast_collate_fn, pin_memory=True)
        test_dataloader_collection.append(test_dataloader)
        query_names_len_collection.append(len(query_names))
        test_names_collection.append(query_names+gallery_names)
    
    return test_dataloader_collection, query_names_len_collection, test_names_collection


def estimate_logits(
    cfg, 
    model, 
    data_loader,
    sample_number=None
):

    model.eval()
    savedir = os.path.join(cfg.OUTPUT_DIR, '-'.join(cfg.DATASETS.TEST_NAMES), 'estimate_logits')
    if not os.path.exists(savedir): os.makedirs(savedir)

    if len(data_loader) > 10:
        print('Estimate logits on train data...')
        outfile = open(os.path.join(savedir, cfg.DATASETS.TEST_NAMES[0]+'_train.txt'), 'w')
        logits, feats, pids, camids, probs, max_probs, pred_ids = [], [], [], [], [], [], []
        prefetcher = data_prefetcher(data_loader)
        batch = prefetcher.next()
        while batch[0] is not None:
            img, pid, camid = batch
            with torch.no_grad():
                logit, feat = model(img)

            # logits.append(logit)
            prob = F.softmax(logit, dim=1)
            # probs.append(prob)
            max_prob, pred_id = torch.max(prob, dim=1)

            pred_ids.extend(pred_id.cpu().numpy().tolist())
            max_probs.extend(max_prob.cpu().numpy().tolist())
            pids.extend(pid.cpu().numpy().tolist())
            # camids.extend(np.asarray(camid))

            batch = prefetcher.next()
            
        item = 0
        max_sample = 1000000
        if len(pids) > max_sample:
            sample_rand = random.sample(range(len(pids)), max_sample)
            for sample_i in sample_rand:
                outfile.write(f'Item {item:6d} : id = {pids[sample_i]:6d} , pred = {pred_ids[sample_i]:6d} , prob = {max_probs[sample_i]:.4f}\n')
                item += 1
        else:
            for a, b, c in zip(pids, pred_ids, max_probs):
                # print(f'Item {item:6d} : id = {a:4d} , pred = {b:4d} , prob = {c:.4f}')
                outfile.write(f'Item {item:6d} : id = {a:6d} , pred = {b:6d} , prob = {c:.4f}\n')
                item += 1
    
    else:
        idx = -1

        for test_dataset_name, dataloader, num_query in zip(cfg.DATASETS.TEST_NAMES, data_loader, sample_number):
            print(f'Estimate logits on test data of {test_dataset_name}')
            idx += 1
            outfile = open(os.path.join(savedir, test_dataset_name+'_test.txt'), 'w')
            logits, feats, pids, camids, probs, max_probs, pred_ids = [], [], [], [], [], [], []
            prefetcher = data_prefetcher(dataloader)
            batch = prefetcher.next()
            while batch[0] is not None:
                img, pid, camid = batch
                with torch.no_grad():
                    logit, feat = model(img)

                # logits.append(logit)
                prob = F.softmax(logit, dim=1)
                # probs.append(prob)
                max_prob, pred_id = torch.max(prob, dim=1)
                
                pred_ids.extend(pred_id.cpu().numpy().tolist())
                max_probs.extend(max_prob.cpu().numpy().tolist())
                pids.extend(pid.cpu().numpy().tolist())
                # camids.extend(np.asarray(camid))

                batch = prefetcher.next()

            if 'veri' == test_dataset_name:
                pred_ids = pred_ids[1678:]
                pids = pids[1678:]
                max_probs = max_probs[1678:]

            item = 0            
            max_sample = 1000000
            if len(pids) > max_sample:
                sample_rand = random.sample(range(len(pids)), max_sample)
                for sample_i in sample_rand:
                    outfile.write(f'Item {item:6d} : id = {pids[sample_i]:6d} , pred = {pred_ids[sample_i]:6d} , prob = {max_probs[sample_i]:.4f}\n')
                    item += 1
            else:
                for a, b, c in zip(pids, pred_ids, max_probs):
                    # print(f'Item {item:6d} : id = {a:4d} , pred = {b:4d} , prob = {c:.4f}')
                    outfile.write(f'Item {item:6d} : id = {a:6d} , pred = {b:6d} , prob = {c:.4f}\n')
                    item += 1

            
def estimate_distance(
    cfg,
    model,
    data_loader,
    sample_number=None
):

    model.eval()

    savedir = os.path.join(cfg.OUTPUT_DIR, '-'.join(cfg.DATASETS.TEST_NAMES), 'estimate_dist')
    if not os.path.exists(savedir): os.makedirs(savedir)
    if len(data_loader) > 10:
        print('Estimate distance on train data...')
        outfile = open(os.path.join(savedir, cfg.DATASETS.TEST_NAMES[0]+'_train.txt'), 'w')
        feats, pids, camids, pos_dists, mean_dists, neg_dists = [], [], [], [], [], []
        prefetcher = data_prefetcher(data_loader)
        batch = prefetcher.next()
        batchi = 0
        while batch[0] is not None:
            print(f'Extracting feature for batch {batchi}')
            img, pid, camid = batch
            with torch.no_grad():
                logit, feat = model(img)
            feats.append(feat)
            pids.extend(pid.cpu().numpy().tolist())
            camids.extend(np.asarray(camid))

            batch = prefetcher.next()
            batchi += 1
            
        feats = torch.cat(feats, dim=0)
        pids = np.asarray(pids)
        camids = np.asarray(camids)
            
        if cfg.TEST.NORM:
            feats = F.normalize(feats, p=2, dim=1)
        
        batchN = feats.shape[0]//cfg.TEST.IMS_PER_BATCH
        print('batchN : ', batchN+1)
        for batch_i in range(batchN+1):
            print(f'Processing batch : {batch_i}')
            if batch_i != batchN:
                feats_split = feats[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
                pids_split = pids[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
                camids_split = camids[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
            else:
                feats_split = feats[batch_i*cfg.TEST.IMS_PER_BATCH:]
                pids_split = pids[batch_i*cfg.TEST.IMS_PER_BATCH:]
                camids_split = camids[batch_i*cfg.TEST.IMS_PER_BATCH:]

            # cosine distance
            distmat_split = torch.mm(feats_split, feats.t()).cpu().numpy()
            distmat_split = -distmat_split

            for sample_i in range(len(pids_split)):
                dist = -distmat_split[sample_i]
                pid = pids_split[sample_i]
                camid = camids_split[sample_i]

                # remove gallery samples that have the same pid and camid with query
                positive = (pids == pid) & (camids != camid)
                negtive = (pids != pid)
                if not np.any(positive):
                    # this condition is true when query identity does not appear in gallery
                    continue
                dist_pos = dist[positive]
                dist_neg = dist[negtive]
                pos_dists.append(np.max(dist_pos))
                neg_dists.append(np.max(dist_neg))
                mean_dists.append(np.mean(dist_pos))
        
        item = 0
        max_sample = 1000000
        if len(pids) > max_sample:
            sample_rand = random.sample(range(len(pids)), max_sample)
            for sample_i in sample_rand:
                outfile.write(f'Item {item:6d} : id = {pids[sample_i]:6d} , pos_dist = {pos_dists[sample_i]:.6f}, mean_dist = {mean_dists[sample_i]:.6f}, neg_dist = {neg_dists[sample_i]:.6f}\n')
                item += 1
        else:
            for a, b, c, d in zip(pids, pos_dists, mean_dists, neg_dists):
                outfile.write(f'Item {item:6d} : id = {a:6d} , pos_dist = {b:.6f} , mean_dist = {c:.6f}, neg_dist = {d:.6f}\n')
                item += 1

    else:
        idx = -1

        for test_dataset_name, dataloader, num_query in zip(cfg.DATASETS.TEST_NAMES, data_loader, sample_number):
            print(f'Estimate distance on test data of {test_dataset_name}')
            idx += 1
            outfile = open(os.path.join(savedir, test_dataset_name+'_test.txt'), 'w')
            feats, pids, camids, pos_dists, mean_dists, neg_dists = [], [], [], [], [], []
            prefetcher = data_prefetcher(dataloader)
            batch = prefetcher.next()
            batchi = 0
            while batch[0] is not None:
                print(f'Extracting feature for batch {batchi}')
                img, pid, camid = batch
                with torch.no_grad():
                    logit, feat = model(img)
                feats.append(feat)
                pids.extend(pid.cpu().numpy().tolist())
                camids.extend(np.asarray(camid))

                batch = prefetcher.next()
                batchi += 1

            feats = torch.cat(feats, dim=0)
            pids = np.asarray(pids)
            camids = np.asarray(camids)

            if 'veri' == test_dataset_name:
                feats = feats[1678:]
                pids = pids[1678:]
                camids = camids[1678:]

            if cfg.TEST.NORM:
                feats = F.normalize(feats, p=2, dim=1)

            batchN = feats.shape[0]//cfg.TEST.IMS_PER_BATCH
            print('batchN : ', batchN+1)
            for batch_i in range(batchN+1):
                print(f'Processing batch : {batch_i}')
                if batch_i != batchN:
                    feats_split = feats[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
                    pids_split = pids[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
                    camids_split = camids[batch_i*cfg.TEST.IMS_PER_BATCH:(batch_i+1)*cfg.TEST.IMS_PER_BATCH]
                else:
                    feats_split = feats[batch_i*cfg.TEST.IMS_PER_BATCH:]
                    pids_split = pids[batch_i*cfg.TEST.IMS_PER_BATCH:]
                    camids_split = camids[batch_i*cfg.TEST.IMS_PER_BATCH:]

                # cosine distance
                distmat_split = torch.mm(feats_split, feats.t()).cpu().numpy()
                distmat_split = -distmat_split

                for sample_i in range(len(pids_split)):
                    dist = -distmat_split[sample_i]
                    pid = pids_split[sample_i]
                    camid = camids_split[sample_i]

                    # remove gallery samples that have the same pid and camid with query
                    positive = (pids == pid) & (camids != camid)
                    negtive = (pids != pid)
                    if not np.any(positive):
                        # this condition is true when query identity does not appear in gallery
                        continue
                    dist_pos = dist[positive]
                    dist_neg = dist[negtive]
                    pos_dists.append(np.max(dist_pos))
                    neg_dists.append(np.max(dist_neg))
                    mean_dists.append(np.mean(dist_pos))

                   
            item = 0
            max_sample = 1000000
            if len(pids) > max_sample:
                sample_rand = random.sample(range(len(pids)), max_sample)
                for sample_i in sample_rand:
                    outfile.write(f'Item {item:6d} : id = {pids[sample_i]:6d} , pos_dist = {pos_dists[sample_i]:.6f}, mean_dist = {mean_dists[sample_i]:.6f}, neg_dist = {neg_dists[sample_i]:.6f}\n')
                    item += 1
            else:
                for a, b, c, d in zip(pids, pos_dists, mean_dists, neg_dists):
                    outfile.write(f'Item {item:6d} : id = {a:6d} , pos_dist = {b:.6f} , mean_dist = {c:.6f}, neg_dist = {d:.6f}\n')
                    item += 1

            
def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument('-cfg',
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    gpus = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else '0'
    gpus = [int(i) for i in gpus.split(',')]
    num_gpus = len(gpus)

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set pretrian = False to avoid loading weight repeatedly
    cfg.MODEL.PRETRAIN = False
    cfg.freeze()

    logger = setup_logger("reid_baseline", False, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True


    train_dataloader, num_classes = get_train_dataloader(cfg)
    test_dataloader_collection, num_query_collection, _ = get_test_dataloader(cfg)

    model = build_model(cfg, num_classes)
    model.load_params_w_fc(torch.load(cfg.TEST.WEIGHT))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

#     estimate_logits(cfg, model, train_dataloader)
#     estimate_logits(cfg, model, test_dataloader_collection, sample_number=num_query_collection)
#     estimate_distance(cfg, model, train_dataloader)
    estimate_distance(cfg, model, test_dataloader_collection, sample_number=num_query_collection)

if __name__ == '__main__':
    main()

