# Experiment baseline : 256x256-bs32x4-warmup10-erase0_5
# Dataset: veri
# imagesize: 256x256
# batchsize: 64x4
# warmup_step 0
# random erase prob 0.5
CUDA_VISIBLE_DEVICES='0,1,2,3' python tools/train.py -cfg='configs/softmax_triplet_veri.yml' \
    DATASETS.NAMES '("veri",)' \
    SOLVER.IMS_PER_BATCH '512' \
    MODEL.WITH_IBN 'False' \
    MODEL.BACKBONE 'resnet50' \
    SOLVER.OPT 'adam' \
    SOLVER.LOSSTYPE '("softmax", "triplet")' \
    OUTPUT_DIR '/home/liuxinchen3/notespace/project/vehiclereid/reid_baseline/experiment/veri/Experiment-baseline-veri_0417_0000'