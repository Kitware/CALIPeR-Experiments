#!/bin/bash
# Legacy: reproduces the BRC1 ground_range protocol (crop-level kwcoco splits read by
# data/datasets/kitbrc1.py), NOT the paper's results -- for those use ./extract_features.sh.
#
# Video-averaging ReID on KITBRC1 ground-range (one mean-pooled feature per video).
#   Per-crop features -> average-pool per video -> video-to-video match.
#   Reports mAP/Rank-k PER query camera (500m, 1000m) and combined, vs the fixed 380m gallery.
#   Model: Market1501 ResNet50 (all tricks), after-BN + cosine.

cd "$(dirname "$0")"

python3 tools/eval_caliper_tracklet.py --config_file='configs/softmax_triplet_with_center.yml' \
    --split ground_range \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.NECK_FEAT "('after')" \
    TEST.FEAT_NORM "('yes')" \
    TEST.WEIGHT "('./weights/market_resnet50_model_120_rank1_945.pth')" \
    OUTPUT_DIR "('./crosscheck_video')"
