# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import numpy as np
import math
import random

from .backbones import *
from .losses.cosface import AddMarginProduct
from .utils import *


# Changed by Xinchen Liu


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj_size=9, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_size = adj_size

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        #self.bn = nn.BatchNorm2d(self.out_features)
        self.bn = nn.BatchNorm1d(out_features * adj_size)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output_ = torch.bmm(adj, support)
        if self.bias is not None:
            output_ =  output_ + self.bias
        output = output_.view(output_.size(0), output_.size(1)*output_.size(2))
        output = self.bn(output)
        output = output.view(output_.size(0), output_.size(1), output_.size(2))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, adj_size, nfeat, nhid, isMeanPooling = True):
        super(GCN, self).__init__()

        self.adj_size = adj_size
        self.nhid = nhid
        self.isMeanPooling = isMeanPooling
        self.gc1 = GraphConvolution(nfeat, nhid ,adj_size)
        self.gc2 = GraphConvolution(nhid, nhid, adj_size)

    def forward(self, x, adj):
        x_ = F.dropout(x, 0.5, training=self.training) 
        x_ = F.relu(self.gc1(x_, adj))
        x_ = F.dropout(x_, 0.5, training=self.training)
        x_ = F.relu(self.gc2(x_, adj))

        x_mean = torch.mean(x_, 1) # aggregate features of nodes by mean pooling
        x_cat = x_.view(x.size()[0], -1) # aggregate features of nodes by concatenation
        x_mean = F.dropout(x_mean, 0.5, training=self.training)
        x_cat = F.dropout(x_cat, 0.5, training=self.training)
        
        return x_mean, x_cat


class Baseline_SelfGCN(nn.Module):
    gap_planes = 2048

    def __init__(self, 
                 backbone, 
                 num_classes,
                 num_parts,
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb, 
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
            self.base_gcn = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)
            self.base_gcn.load_pretrain(model_path)
            
        self.gcn = GCN(num_parts-1, self.gap_planes, self.gap_planes, isMeanPooling = True)        
        self.num_classes = num_classes
        self.num_parts = num_parts # 1 for only foreground, 10 for masks of ten parts
        
        # Global Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Global head
        self.bottleneck = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.gap_planes, self.num_classes, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # GCN head
        self.bottleneck_gcn = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck_gcn.bias.requires_grad_(False)  # no shift
        self.classifier_gcn = nn.Linear(self.gap_planes, self.num_classes, bias=False)
        self.bottleneck_gcn.apply(weights_init_kaiming)
        self.classifier_gcn.apply(weights_init_classifier)
        
    def forward(self, inputs_global, inputs_gcn, mask, adj):
        # Global Branch
        x_global = self.base(inputs_global)

        feat_global = self.gap(x_global)  # (b, 2048, 1, 1)
        feat_global = feat_global.view(-1, feat_global.size()[1])
        bnfeat_global = self.bottleneck(feat_global)  # normalize for angular softmax

        
        # Self-GCN Branch
        x_gcn = self.base_gcn(inputs_gcn)

        h, w = x_gcn.size(2), x_gcn.size(3)
        mask_resize = F.interpolate(input=mask.float(), size=(h, w), mode='nearest')
        # random part drop
        x_self_list = list()
        for i in range(x_gcn.size(0)): # randomly drop one part for each sample
            mask_self = mask_resize[i]
            part_list = []
            for c in range(1, self.num_parts):
                part = (mask_self.long() == c)
                if part.any():
                    part_list.append(c)
            drop_part = random.choice(part_list)
            mask_self = (mask_self.long() != drop_part)
            x_self = mask_self.float()*x_gcn[i]
            x_self = x_self.unsqueeze(0)
            x_self_list.append(x_self)
        x_self = torch.cat(x_self_list, dim=0)

        mask_list = list()
        mask_list.append((mask_resize.long() > 0))
        for c in range(1, self.num_parts):
            mask_list.append((mask_resize.long() == c)) # split mask of each class

        x_list = list()
        x_self_list = list()
        for c in range(self.num_parts):
            x_list.append(mask_list[c].float() * x_gcn) # split feature map by mask of each class
            x_self_list.append(mask_list[c].float() * x_self)

        for c in range(1, self.num_parts):
            x_list[c] = (x_list[c].sum(dim=2).sum(dim=2)) / \
                        (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
            x_list[c] = x_list[c].unsqueeze(1) # keep 2048
            
            x_self_list[c] = (x_self_list[c].sum(dim=2).sum(dim=2)) / \
                        (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
            x_self_list[c] = x_self_list[c].unsqueeze(1) # keep 2048

        mask_feat = torch.cat(x_list[1:], dim=1) # concat all parts to feat matrix b*part*feat
        self_feat = torch.cat(x_self_list[1:], dim=1)

        feat_gcn_mean, feat_gcn_cat = self.gcn(mask_feat, adj) # feat*9 to feat by gcn
        feat_gcn = feat_gcn_mean.view(-1, feat_gcn_mean.size()[1])
        feat_gcn_cat = feat_gcn_cat.view(-1, feat_gcn_cat.size()[1])
        
        feat_self_mean, feat_self_cat = self.gcn(self_feat, adj) # feat*9 to feat by gcn
        feat_self = feat_self_mean.view(-1, feat_self_mean.size()[1])
        feat_self_cat = feat_self_cat.view(-1, feat_self_cat.size()[1])

        bnfeat_gcn = self.bottleneck_gcn(feat_gcn)
        bnfeat_self = self.bottleneck_gcn(feat_self)
        
        if self.training:
            cls_score = self.classifier(bnfeat_global)
            cls_score_gcn = self.classifier_gcn(bnfeat_gcn)
            cls_score_self = self.classifier_gcn(bnfeat_self)
            return cls_score, feat_global, cls_score_gcn, bnfeat_gcn, cls_score_self, bnfeat_self, feat_gcn_cat, feat_self_cat
#             return cls_score, feat_global, cls_score_gcn, feat_gcn, cls_score_self, feat_self, feat_gcn_cat, feat_self_cat
        else:
            cls_score = None
            cls_score_gcn = None
            cls_score_self = None
            return cls_score, bnfeat_global, cls_score_gcn, bnfeat_gcn, cls_score_self, bnfeat_self, feat_gcn_cat, feat_self_cat

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier_gcn.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)

    def load_params_w_fc(self, state_dict):
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)

