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
#         x_ = F.dropout(x_, 0.5, training=self.training)
        x_mean = F.dropout(x_mean, 0.5, training=self.training)
        x_cat = F.dropout(x_cat, 0.5, training=self.training)
        
        return x_mean, x_cat


class Baseline(nn.Module):
    gap_planes = 2048

    def __init__(self, 
                 backbone, 
                 num_classes, 
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb, 
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        
        self.bottleneck = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.gap_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(-1, global_feat.size()[1])
        bnfeat = self.bottleneck(global_feat)  # normalize for angular softmax
        cls_score = self.classifier(bnfeat)
        if self.training:
            # cls_score = self.classifier(feat, label)
            return cls_score, global_feat
        else:
            return cls_score, bnfeat

    def load_params_wo_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def load_params_w_fc(self, state_dict):
        # new_state_dict = {}
        # for k, v in state_dict.items():
        #     k = '.'.join(k.split('.')[1:])
        #     new_state_dict[k] = v
        # state_dict = new_state_dict
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)


# Changed by Xinchen Liu

class Baseline_GCN(nn.Module):
    gap_planes = 2048
    part_planes = 256

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
        # Mask Branch
#         self.part_share_fc = nn.Linear(self.gap_planes, self.part_planes, bias=False)
#         self.fusion_fc = nn.Linear(self.part_planes*self.num_parts, self.gap_planes, bias=False)
        
        # Global Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fusion of Global and Mask
        self.embedding = nn.Linear(self.gap_planes*2, self.gap_planes, bias=False)
        
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

#         # Local head
#         self.localmax = True
#         if self.localmax:
#             self.bottleneck_l = nn.BatchNorm1d(self.gap_planes)
#             self.bottleneck_l.bias.requires_grad_(False)  # no shift
#             self.classifier_l = nn.Linear(self.gap_planes, self.num_classes, bias=False)
#         else:
#             self.bottleneck_l = nn.BatchNorm1d(self.gap_planes*9)
#             self.bottleneck_l.bias.requires_grad_(False)  # no shift
#             self.classifier_l = nn.Linear(self.gap_planes*9, self.num_classes, bias=False)

#         self.bottleneck_l.apply(weights_init_kaiming)
#         self.classifier_l.apply(weights_init_classifier)
        
    def forward(self, x, mask, adj, use_mask=True): # use image + mask by concatenate
        x_global = self.base(x)
        x_gcn = self.base_gcn(x)

        if use_mask: # Mask Branch
            h, w = x_gcn.size(2), x_gcn.size(3)
            mask_resize = F.interpolate(input=mask.float(), size=(h, w), mode='nearest')
            mask_list = list()
            mask_list.append((mask_resize.long() > 0))
            for c in range(1, self.num_parts):
                mask_list.append((mask_resize.long() == c)) # split mask of each class
           
            x_list = list()
            for c in range(self.num_parts):
                x_list.append(mask_list[c].float() * x_gcn) # split feature map by mask of each class
            
            if self.num_parts > 1:
                for c in range(1, self.num_parts):
                    x_list[c] = (x_list[c].sum(dim=2).sum(dim=2)) / \
                                (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
                    x_list[c] = x_list[c].unsqueeze(1) # keep 2048
                    #x_list[c] = self.part_share_fc(x_list[c]).unsqueeze(1) # 2048 to 256 by fc

                mask_feat = torch.cat(x_list[1:], dim=1) # concat all parts to feat matrix b*part*feat

#                 if self.localmax:
#                     feat_local = torch.max(mask_feat, dim=1)[0] # max pooling of parts
#                 else:
#                     feat_local = mask_feat.view(mask_feat.size(0), -1) # concatenate of parts
#                 feat_local = feat_local.view(-1, feat_local.size()[1])

                feat_gcn, _ = self.gcn(mask_feat, adj) # feat*9 to feat by gcn
                feat_gcn = feat_gcn.view(-1, feat_gcn.size()[1])

            else:
                mask_feat = (x_list[0].sum(dim=2).sum(dim=2)) / \
                    (mask_list[0].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8)
                mask_feat = mask_feat.view(-1, mask_feat.size()[1]) # (b, 2048, 1, 1) to (b, 2048)

        # Global Branch
        feat_global = self.gap(x_global)  # (b, 2048, 1, 1)
        feat_global = feat_global.view(-1, feat_global.size()[1])
        feat_global = self.bottleneck(feat_global)  # normalize for angular softmax

        feat_gcn = self.bottleneck_gcn(feat_gcn)
#         feat_local = self.bottleneck_l(feat_local)
        
        if self.training:
            cls_score = self.classifier(feat_global)
            cls_score_gcn = self.classifier_gcn(feat_gcn)
#             cls_score_local = self.classifier_l(feat_local)
            return cls_score, feat_global, cls_score_gcn, feat_gcn, cls_score_gcn, feat_gcn
        else:
            cls_score = None
            cls_score_gcn = None
#             cls_score_local = None
            return cls_score, feat_global, cls_score_gcn, feat_gcn, cls_score_gcn, feat_gcn
        
    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier_gcn.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def load_params_w_fc(self, state_dict):
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)




class Baseline_Mask(nn.Module):
    gap_planes = 2048
    part_planes = 256

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
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)
        self.num_classes = num_classes
        self.num_parts = num_parts # 1 for only foreground, 10 for masks of ten parts
        # Mask Branch
        self.part_share_fc = nn.Linear(self.gap_planes, self.part_planes, bias=False)
        self.fusion_fc = nn.Linear(self.part_planes*self.num_parts, self.gap_planes, bias=False)
        
        # Global Branch
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fusion of Global and Mask
        self.embedding = nn.Linear(self.gap_planes*2, self.gap_planes, bias=False)
        
        self.bottleneck = nn.BatchNorm1d(self.gap_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.gap_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, mask, use_mask=True): # use image or masked image
        x = self.base(x)
        use_mask=False
        
        if use_mask: # Mask Branch
            h, w = x.size(2), x.size(3)
            mask_resize = F.interpolate(input=mask.float(), size=(h, w), mode='nearest')
            mask_list = list()
            mask_list.append((mask_resize.long() > 0))
            for c in range(1, self.num_parts):
                mask_list.append((mask_resize.long() == c)) # split mask of each class
           
            x_list = list()
            for c in range(self.num_parts):
                x_list.append(mask_list[c].float() * x) # split feature map by mask of each class
            
            if self.num_parts > 1:
                for c in range(self.num_parts):
                    x_list[c] = (x_list[c].sum(dim=2).sum(dim=2)) / \
                                (mask_list[c].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8) # GAP feature of each part
                    x_list[c] = self.part_share_fc(x_list[c]) # 2048 to 256 for each part

                mask_feat = torch.cat(x_list, dim=1) # concat all parts by channel
                mask_feat = self.fusion_fc(mask_feat) # 256*10 to 2048
            else:
                mask_feat = (x_list[0].sum(dim=2).sum(dim=2)) / \
                    (mask_list[0].squeeze(dim=1).sum(dim=1).sum(dim=1).float().unsqueeze(dim=1)+1e-8)
            
            mask_feat = mask_feat.view(-1, mask_feat.size()[1]) # (b, 2048, 1, 1) to (b, 2048)
        else: # Global Branch
            global_feat = self.gap(x)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(-1, global_feat.size()[1])
        
        if use_mask:
#             g_m_feat = torch.cat((mask_feat, global_feat), dim=1)
#             g_m_feat = self.embedding(g_m_feat)
            g_m_feat = mask_feat
        else:
            g_m_feat = global_feat

        feat = self.bottleneck(g_m_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, feat
        else:
            cls_score = None
            return cls_score, feat

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)
        # assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def load_params_w_fc(self, state_dict):
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)


class Baseline_InsDis(nn.Module):
    gap_planes = 2048
    in_planes = 128 # add for InsDis

    def __init__(self, 
                 backbone, 
                 num_classes, 
                 last_stride, 
                 with_ibn, 
                 gcb, 
                 stage_with_gcb, 
                 pretrain=True, 
                 model_path=''):
        super().__init__()
        try:
            self.base = ResNet.from_name(backbone, last_stride, with_ibn, gcb, stage_with_gcb)
        except:
            print(f'not support {backbone} backbone')

        if pretrain:
            self.base.load_pretrain(model_path)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.emb = nn.Linear(self.gap_planes, self.in_planes, bias=False) # add for InsDis
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        
        self.l2norm = Normalize(2) # add for InsDis

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(-1, global_feat.size()[1])
        
        emb_feat = self.emb(global_feat) # 2048 -> 128 add for InsDis
        
        feat = self.bottleneck(emb_feat)  # normalize for angular softmax
        if self.training:
            cls_score = self.classifier(feat)
            feat = self.l2norm(feat) # add for InsDis
            # cls_score = self.classifier(feat, label)
            return cls_score, feat
        else:
            return feat

    def load_params_wo_fc(self, state_dict):
        state_dict.pop('classifier.weight')
        res = self.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(['classifier.weight',]), 'issue loading pretrained weights'

    def load_params_w_fc(self, state_dict):
        res = self.load_state_dict(state_dict, strict=False)
        print("Loading Pretrained Model ... Missing Keys: ", res.missing_keys)
