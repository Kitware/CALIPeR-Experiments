#!/bin/bash
# Image-level ReID on the CALIPeR KITBRC1 ground-range benchmark (score each crop/image independently).
#   Fixed 380m gallery vs combined 500m+1000m query.
#   Model: Market1501 ResNet50 (all tricks), after-BN + cosine (matches the checkpoint's reported 94.5%).
#
# KITBRC1_MAX_PER_APPEARANCE=16 -> <=16 crops per (video,subject); keeps the query x gallery distance
#                             matrix small.
#
# NOTE: query side (esp. 1000m) is preliminary -- the upstream best-track heuristic is unsupervised, so the
#       selected track isn't guaranteed to be the labeled subject; numbers improve with subject-track GT.

cd "$(dirname "$0")"

KITBRC1_SPLIT=ground_range \
KITBRC1_MAX_PER_APPEARANCE=16 \
python3 tools/test.py --config_file='configs/softmax_triplet_with_center.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('kitbrc1')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.NECK_FEAT "('after')" \
    TEST.FEAT_NORM "('yes')" \
    TEST.WEIGHT "('./weights/market_resnet50_model_120_rank1_945.pth')" \
    OUTPUT_DIR "('./crosscheck_image')"
