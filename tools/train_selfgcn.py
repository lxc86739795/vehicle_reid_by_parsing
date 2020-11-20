# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# Changed by Xinchen Liu

import argparse
import os
import sys

from torch.backends import cudnn

sys.path.append(".")
from config import cfg
from utils.logger import setup_logger
from engine.trainer_selfgcn import ReidSystem
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="ReID Model Training")
    parser.add_argument(
        '-cfg', "--config_file", 
        default="", 
        metavar="FILE", 
        help="path to config file", 
        type=str
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    gpus = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else '0'
    gpus = [int(i) for i in gpus.split(',')]
    num_gpus = len(gpus)

    cfg.freeze()

    log_save_dir = os.path.join(cfg.OUTPUT_DIR, '-'.join(cfg.DATASETS.TEST_NAMES), cfg.MODEL.VERSION)
    if not os.path.exists(log_save_dir): os.makedirs(log_save_dir)

    logger = setup_logger("reid_baseline.train", log_save_dir, 0)
    logger.info("Using {} GPUs.".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('start training')
    cudnn.benchmark = True

    writer = SummaryWriter(os.path.join(log_save_dir, 'tf'))
    reid_system = ReidSystem(cfg, logger, writer)
    reid_system.train()

if __name__ == '__main__':
    main()
