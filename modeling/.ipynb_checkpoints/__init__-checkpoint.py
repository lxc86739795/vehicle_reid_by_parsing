# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from torch import nn

from .baseline import Baseline, Baseline_InsDis, Baseline_Mask, Baseline_GCN
from .baseline_selfgcn import Baseline_SelfGCN
from .losses import reidLoss

# Changed by Xinchen Liu

def build_model(cfg, num_classes, use_mask=False) -> nn.Module:
    if 'InsDis' in list(cfg.SOLVER.LOSSTYPE):
        print('Baseline Instance Model')
        model = Baseline_InsDis(
            cfg.MODEL.BACKBONE, 
            num_classes, 
            cfg.MODEL.LAST_STRIDE, 
            cfg.MODEL.WITH_IBN, 
            cfg.MODEL.GCB, 
            cfg.MODEL.STAGE_WITH_GCB, 
            cfg.MODEL.PRETRAIN, 
            cfg.MODEL.PRETRAIN_PATH)
    elif use_mask:
        print('Baseline with Mask Branch')
        model = Baseline_Mask(
            cfg.MODEL.BACKBONE, 
            num_classes,
            cfg.MODEL.NUM_PARTS,
            cfg.MODEL.LAST_STRIDE, 
            cfg.MODEL.WITH_IBN, 
            cfg.MODEL.GCB, 
            cfg.MODEL.STAGE_WITH_GCB, 
            cfg.MODEL.PRETRAIN, 
            cfg.MODEL.PRETRAIN_PATH)
    else:
        print('Baseline Model')
        model = Baseline(
            cfg.MODEL.BACKBONE, 
            num_classes, 
            cfg.MODEL.LAST_STRIDE, 
            cfg.MODEL.WITH_IBN, 
            cfg.MODEL.GCB, 
            cfg.MODEL.STAGE_WITH_GCB, 
            cfg.MODEL.PRETRAIN, 
            cfg.MODEL.PRETRAIN_PATH)
    return model

def build_model_gcn(cfg, num_classes, use_mask=False) -> nn.Module:
    print('Baseline GCN Model')
    model = Baseline_GCN(
        cfg.MODEL.BACKBONE, 
        num_classes,
        cfg.MODEL.NUM_PARTS,
        cfg.MODEL.LAST_STRIDE, 
        cfg.MODEL.WITH_IBN, 
        cfg.MODEL.GCB, 
        cfg.MODEL.STAGE_WITH_GCB, 
        cfg.MODEL.PRETRAIN, 
        cfg.MODEL.PRETRAIN_PATH)

    return model

def build_model_selfgcn(cfg, num_classes) -> nn.Module:
    print('Baseline SelfGCN Model')
    model = Baseline_SelfGCN(
        cfg.MODEL.BACKBONE, 
        num_classes,
        cfg.MODEL.NUM_PARTS,
        cfg.MODEL.LAST_STRIDE, 
        cfg.MODEL.WITH_IBN, 
        cfg.MODEL.GCB, 
        cfg.MODEL.STAGE_WITH_GCB, 
        cfg.MODEL.PRETRAIN, 
        cfg.MODEL.PRETRAIN_PATH)

    return model
