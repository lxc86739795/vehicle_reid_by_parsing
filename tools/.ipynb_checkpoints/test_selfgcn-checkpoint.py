# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import torch
from torch.backends import cudnn
from torch import nn

# Changed by Xinchen Liu

sys.path.append('.')
from config import cfg
from data import get_test_dataloader_mask
from engine.inference_selfgcn import inference
from modeling import build_model_selfgcn
from utils.logger import setup_logger


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

    model = build_model_selfgcn(cfg, 0)
    model.load_params_wo_fc(torch.load(cfg.TEST.WEIGHT))
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.cuda()

    print('prepare test set ...')
    test_dataloader_collection, num_query_collection, _ = get_test_dataloader_mask(cfg)

    inference(cfg, model, test_dataloader_collection, num_query_collection, use_mask=True)


if __name__ == '__main__':
    main()

