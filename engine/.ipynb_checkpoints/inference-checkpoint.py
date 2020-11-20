# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging
import os
import torch
import numpy as np
import torch.nn.functional as F
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher, data_prefetcher_mask
import random
import cv2 

# Changed by Xinchen Liu

def inference(
        cfg,
        model,
        test_dataloader_collection,
        num_query_collection,
        is_vis=False,
        test_collection=None,
        use_mask=True,
        num_parts=1,
        mask_image=True
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    idx = -1
    for test_dataset_name, test_dataloader, num_query in zip(cfg.DATASETS.TEST_NAMES, test_dataloader_collection, num_query_collection):
        idx += 1
        feats, pids, camids = [], [], []
        if use_mask:
            test_prefetcher = data_prefetcher_mask(test_dataloader)
        else:
            test_prefetcher = data_prefetcher(test_dataloader)
        batch = test_prefetcher.next()
        while batch[0] is not None:
            if use_mask:
                img, mask, pid, camid = batch
                if mask_image:
                    h, w = img.size(2), img.size(3)
                    assert img.size(2) == mask.size(2) and img.size(3) == mask.size(3)
                    mask_list = list()
                    mask_list.append((mask.long() > 0))
                    for c in range(1, num_parts):
                        mask_list.append((mask_resize.long() == c)) # split mask of each class
                    x_list = list()
                    for c in range(num_parts):
                        x_list.append(mask_list[c].float() * img) # segment input by mask of each class
                    img = x_list[0]
                with torch.no_grad():
                    _, feat = model(img, mask, use_mask=False)
            else:
                img, pid, camid = batch
                with torch.no_grad():
                    _, feat = model(img)

            feats.append(feat)
            pids.extend(pid.cpu().numpy())
            camids.extend(np.asarray(camid))

            batch = test_prefetcher.next()

        feats = torch.cat(feats, dim=0)
        if cfg.TEST.NORM:
            feats = F.normalize(feats, p=2, dim=1)
        # query
        qf = feats[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf = feats[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])

        # cosine distance
        distmat = torch.mm(qf, gf.t()).cpu().numpy()

        # euclidean distance
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.numpy()
        cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
        logger.info(f"Results on {test_dataset_name} : ")
        logger.info(f"mAP: {mAP:.1%}")
        for r in [1, 5, 10]:
            logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
            
        if is_vis:
            query_rand = 10
            topK = 10
            is_save_all = True
            query_rand_idx = range(0, num_query) if is_save_all else random.sample(range(0, num_query), query_rand)
            print(f'|-------- Randomly saving top-{topK} results of {len(query_rand_idx)} queries for {test_dataset_name} --------|')
            
            qf_rand = qf[query_rand_idx]
            q_pids_rand = q_pids[query_rand_idx]
            q_camids_rand = q_camids[query_rand_idx]
            
            q_items = test_collection[idx][:num_query]
            q_items_rand = list()
            for i in query_rand_idx:
                q_items_rand.append(q_items[i])
            g_items = test_collection[idx][num_query:]
            
            distmat_rand = torch.mm(qf_rand, gf.t()).cpu().numpy()
            distmat_rand = -distmat_rand
            
            indices = np.argsort(distmat_rand, axis=1)
            matches = (g_pids[indices] == q_pids_rand[:, np.newaxis]).astype(np.int32)
            
            save_img_size = (256, 256)
            
            if test_dataset_name == 'market1501':
                save_img_size = (128, 256)
            
            for q_idx in range(len(query_rand_idx)):
                savefilename = ''
                # get query pid and camid
                q_path = q_items_rand[q_idx][0]
                q_pid = q_items_rand[q_idx][1]
                q_camid = q_items_rand[q_idx][2]
                
                savefilename += 'q-'+q_path.split('/')[-1]+'_g'

                # remove gallery samples that have the same pid and camid with query
                order = indices[q_idx]
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                keep = np.invert(remove)

                print('Query Path : ', q_path)
                print('Result idx : ', order[:topK])
                
                img_list = list()
                q_img = cv2.imread(q_path)
                q_img = cv2.resize(q_img, save_img_size)
                cv2.rectangle(q_img, (0,0), save_img_size, (255,0,0), 4)
                img_list.append(q_img)
                
                for g_idx in order[:topK]:
                    g_img = cv2.imread(g_items[g_idx][0])
                    g_img = cv2.resize(g_img, save_img_size)
                    if q_pid == g_items[g_idx][1] and q_camid == g_items[g_idx][2]:
                        cv2.rectangle(g_img, (0,0), save_img_size, (255,255,0), 4)
                    elif q_pid == g_items[g_idx][1] and q_camid != g_items[g_idx][2]:
                        cv2.rectangle(g_img, (0,0), save_img_size, (0,255,0), 4)
                    else:
                        cv2.rectangle(g_img, (0,0), save_img_size, (0,0,255), 4)
                    img_list.append(g_img)
                    savefilename += '-'+str(g_items[g_idx][1])
                
                pic = np.concatenate(img_list, 1)
                picsavedir = os.path.join(cfg.OUTPUT_DIR, '-'.join(cfg.DATASETS.TEST_NAMES), 'examples', test_dataset_name)
                if not os.path.exists(picsavedir): os.makedirs(picsavedir)
                savefilepath = os.path.join(picsavedir, savefilename+'.jpg')
                cv2.imwrite(savefilepath, pic)
                print('Save example picture to ', savefilepath)
