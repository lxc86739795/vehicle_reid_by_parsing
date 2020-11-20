# Experiment baseline : 256x256-bs32x4-warmup10-erase0_5
# Dataset: veri
# imagesize: 256x256
# batchsize: 32x4
# warmup_step 0
# random erase prob 0
CUDA_VISIBLE_DEVICES='0,1,2,3' python tools/train_selfgcn.py -cfg='configs/softmax_triplet_veri_mask.yml' \
    DATASETS.NAMES '("veri_mask",)' \
    DATASETS.TEST_NAMES '("veri_mask",)' \
    SOLVER.IMS_PER_BATCH '256' \
    SOLVER.OPT 'adam' \
    SOLVER.LOSSTYPE '("softmax", "triplet")' \
    MODEL.BACKBONE 'resnet50' \
    MODEL.PRETRAIN_PATH '/home/liuxinchen3/notespace/project/resnet_pretrain/resnet50-19c8e357.pth' \
    MODEL.NUM_PARTS '10' \
    OUTPUT_DIR '/home/liuxinchen3/notespace/project/vehiclereid/reid_baseline/final_experiment/veri_mask/Experiment-2branch_global_selfgcn_0421_0000'