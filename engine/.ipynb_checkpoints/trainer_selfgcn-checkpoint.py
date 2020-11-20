# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import shutil
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# Changed by Xinchen Liu

from data import get_dataloader_mask
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher_mask
from modeling import build_model_selfgcn
from modeling.losses import TripletLoss
from solver.build import make_lr_scheduler, make_optimizer
from utils.meters import AverageMeter


def L_Matrix(adj_npy, adj_size):

    D =np.zeros((adj_size, adj_size))
    for i in range(adj_size):
        tmp = adj_npy[i,:]
        count = np.sum(tmp==1)
        if count>0:
            number = count ** (-1/2)
            D[i,i] = number

    x = np.matmul(D,adj_npy)
    L = np.matmul(x,D)
    return L


coarse_adj_list = [
    # 1  2  3  4  5  6  7  8  9
    [ 1, 1, 0, 1, 0, 1, 0, 1, 0], #1
    [ 1, 1, 1, 1, 0, 1, 0, 0, 0], #2
    [ 0, 1, 1, 0, 1, 0, 1, 0, 0], #3
    [ 1, 1, 0, 1, 1, 0, 0, 1, 0], #4
    [ 0, 0, 1, 1, 1, 0, 0, 0, 1], #5
    [ 1, 1, 0, 0, 0, 1, 1, 1, 0], #6
    [ 0, 0, 1, 0, 0, 1, 1, 0, 1], #7
    [ 1, 0, 0, 1, 0, 1, 0, 1, 1], #8
    [ 0, 0, 0, 0, 1, 0, 1, 1, 1]  #9
]
coarse_adj_npy = np.array(coarse_adj_list)
coarse_adj_npy = L_Matrix(coarse_adj_npy, len(coarse_adj_npy))


class ReidSystem():
    def __init__(self, cfg, logger, writer):
        self.cfg, self.logger, self.writer = cfg, logger, writer
        # Define dataloader
        self.tng_dataloader, self.val_dataloader_collection, self.num_classes, self.num_query_len_collection = get_dataloader_mask(cfg)
        # networks
        self.use_part_erasing = False
        self.num_parts = cfg.MODEL.NUM_PARTS
        self.model = build_model_selfgcn(cfg, self.num_classes)
        self.adj = torch.from_numpy(coarse_adj_npy).float()
        # loss function
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet = TripletLoss(cfg.SOLVER.MARGIN)
        self.mse_loss = nn.MSELoss()
        # optimizer and scheduler
        self.opt = make_optimizer(self.cfg, self.model)
        self.lr_sched = make_lr_scheduler(self.cfg, self.opt)

        self.loss_weight = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.4]
        self.logger.info(f"Loss weights: {self.loss_weight}, use_pe: {self.use_part_erasing}, use_bnfeat: {True}")
        self._construct()

    def _construct(self):
        self.global_step = 0
        self.current_epoch = 0
        self.batch_nb = 0
        self.max_epochs = self.cfg.SOLVER.MAX_EPOCHS
        self.log_interval = self.cfg.SOLVER.LOG_INTERVAL
        self.eval_period = self.cfg.SOLVER.EVAL_PERIOD
        self.use_dp = False
        self.use_ddp = False

    def loss_fns(self, outputs, labels_global, labels_gcn):
        loss_dict = {}

        if 'softmax' in list(self.cfg.SOLVER.LOSSTYPE):
            loss_dict['ce_g'] = self.ce_loss(outputs[0], labels_global)*self.loss_weight[0]
            loss_dict['ce_l1'] = self.ce_loss(outputs[2], labels_gcn)*self.loss_weight[2]
            loss_dict['ce_l2'] = self.ce_loss(outputs[4], labels_gcn)*self.loss_weight[4]
        if 'triplet' in list(self.cfg.SOLVER.LOSSTYPE):
            loss_dict['tr_g'] = self.triplet(outputs[1], labels_global)[0]*self.loss_weight[1]
            loss_dict['tr_l1'] = self.triplet(outputs[3], labels_gcn)[0]*self.loss_weight[3]
            loss_dict['tr_l2'] = self.triplet(outputs[5], labels_gcn)[0]*self.loss_weight[5]
        
#         target_gcn_feat = outputs[6].clone().detach().requires_grad_(False)
#         loss_dict['mse'] = self.mse_loss(target_gcn_feat, outputs[7])*self.loss_weight[6]
        loss_dict['mse'] = self.mse_loss(outputs[6], outputs[7])*self.loss_weight[6]
        
        return loss_dict

    def on_train_begin(self):
        self.best_mAP = -np.inf
        self.running_loss = AverageMeter()
        log_save_dir = os.path.join(self.cfg.OUTPUT_DIR, '-'.join(self.cfg.DATASETS.TEST_NAMES), self.cfg.MODEL.VERSION)
        self.model_save_dir = os.path.join(log_save_dir, 'ckpts')
        if not os.path.exists(self.model_save_dir): os.makedirs(self.model_save_dir)

        self.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        self.use_dp = (len(self.gpus) > 0) and (self.cfg.MODEL.DIST_BACKEND == 'dp')

        if self.use_dp:
            self.model = nn.DataParallel(self.model)

        self.model = self.model.cuda()
        self.model.train()
        self.adj = self.adj.cuda()

    def on_epoch_begin(self):
        self.batch_nb = 0
        self.current_epoch += 1
        self.t0 = time.time()
        self.running_loss.reset()

        self.tng_prefetcher = data_prefetcher_mask(self.tng_dataloader)

    def training_step(self, batch):

        inputs, masks, labels, _ = batch
        adj_batch = self.adj.repeat(inputs.size(0), 1, 1)
        
        inputs_global = inputs
        inputs_selfgcn = inputs
        labels_global = labels
        labels_selfgcn = labels
        
        if self.use_part_erasing:
#             inputs_masked = torch.zeros(inputs.size(), dtype=inputs.dtype)
            
            # random part erasing
            for i in range(inputs.size(0)):
                input = inputs[i]
                mask = masks[i]
                part_list = []
                for c in range(1, self.num_parts):
                    part = (mask.long() == c)
                    if part.any():
                        part_list.append(c)
                drop_part = random.choice(part_list)
                mask = (mask.long() != drop_part)
                if random.uniform(0, 1) > 0.5:
                    inputs_selfgcn[i] = mask.float()*input
#                 inputs_masked[i] = mask.float()*input
            
#             inputs_masked = inputs_masked.cuda()
#             inputs_global = torch.cat([inputs, inputs_masked], dim=0)
#             labels_global = torch.cat([labels, labels], dim=0)

        outputs = self.model(inputs_global, inputs_selfgcn, masks, adj_batch)

        loss_dict = self.loss_fns(outputs, labels_global, labels_selfgcn)

        total_loss = 0
        print_str = f'\r Epoch {self.current_epoch} Iter {self.batch_nb}/{len(self.tng_dataloader)} '
        for loss_name, loss_value in loss_dict.items():
            total_loss += loss_value
            print_str += (loss_name+f': {loss_value.item():.3f} ')
        loss_dict['total_loss'] = total_loss.item()
        print_str += f'Total loss: {total_loss.item():.3f} '
        print(print_str, end=' ')
        
        if (self.global_step+1) % self.log_interval == 0:
            for loss_name, loss_value in loss_dict.items():
                self.writer.add_scalar(loss_name, loss_value, self.global_step)
            self.writer.add_scalar('total_loss', loss_dict['total_loss'], self.global_step)

        self.running_loss.update(total_loss.item())

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        self.global_step += 1
        self.batch_nb += 1
    
    def on_epoch_end(self):
        elapsed = time.time() - self.t0
        mins = int(elapsed) // 60
        seconds = int(elapsed - mins * 60)
        print('')
        self.logger.info(f'Epoch {self.current_epoch} Total loss: {self.running_loss.avg:.3f} '
                         f'lr: {self.opt.param_groups[0]["lr"]:.2e} During {mins:d}min:{seconds:d}s')
        # update learning rate
        self.lr_sched.step()

    def test(self):
        # convert to eval mode
        self.model.eval()
        
        metric_dict = list()
        for val_dataset_name, val_dataloader, num_query in zip(self.cfg.DATASETS.TEST_NAMES, self.val_dataloader_collection, self.num_query_len_collection):
            feats, pids, camids = [], [], []
            val_prefetcher = data_prefetcher_mask(val_dataloader)
            batch = val_prefetcher.next()
            while batch[0] is not None:
                img, mask, pid, camid = batch
                adj_batch = self.adj.repeat(img.size(0), 1, 1)
                
                with torch.no_grad():
                    output = self.model(img, img, mask, adj_batch)
                
#                 feat = output[1]
                feat = torch.cat([output[1], output[3]], dim=1)
                
                feats.append(feat)
                pids.extend(pid.cpu().numpy())
                camids.extend(np.asarray(camid))

                batch = val_prefetcher.next()

            feats = torch.cat(feats, dim=0)
            if self.cfg.TEST.NORM:
                feats = F.normalize(feats, p=2, dim=1)
            # query
            qf = feats[:num_query]
            q_pids = np.asarray(pids[:num_query])
            q_camids = np.asarray(camids[:num_query])
            # gallery
            gf = feats[num_query:]
            g_pids = np.asarray(pids[num_query:])
            g_camids = np.asarray(camids[num_query:])

            # m, n = qf.shape[0], gf.shape[0]
            distmat = torch.mm(qf, gf.t()).cpu().numpy()
            # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
            # distmat.addmm_(1, -2, qf, gf.t())
            # distmat = distmat.numpy()
            cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
            self.logger.info(f"Test Results on {val_dataset_name} - Epoch: {self.current_epoch}")
            self.logger.info(f"mAP: {mAP:.1%}")
            for r in [1, 5, 10]:
                self.logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")

            self.writer.add_scalar('rank1', cmc[0], self.global_step)
            self.writer.add_scalar('mAP', mAP, self.global_step)
            metric_dict.append({'rank1': cmc[0], 'mAP': mAP})
        # convert to train mode
        self.model.train()
        return metric_dict[0]

    def train(self):
        self.on_train_begin()
#         self.test()
        for epoch in range(self.max_epochs):
            self.on_epoch_begin()
            batch = self.tng_prefetcher.next()
            while batch[0] is not None:
                self.training_step(batch)
                batch = self.tng_prefetcher.next()
            self.on_epoch_end()
            if (epoch+1) % self.eval_period == 0:
                metric_dict = self.test()
                if metric_dict['mAP'] > self.best_mAP:
                    is_best = True
                    self.best_mAP = metric_dict['mAP']
                else:
                    is_best = False
                self.save_checkpoints(is_best)

            torch.cuda.empty_cache()

    def save_checkpoints(self, is_best):
        if self.use_dp:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        
        # TODO: add optimizer state dict and lr scheduler
        filepath = os.path.join(self.model_save_dir, f'model_epoch{self.current_epoch}.pth')
        torch.save(state_dict, filepath)
        if is_best:
            best_filepath = os.path.join(self.model_save_dir, 'model_best.pth')
            shutil.copyfile(filepath, best_filepath)
