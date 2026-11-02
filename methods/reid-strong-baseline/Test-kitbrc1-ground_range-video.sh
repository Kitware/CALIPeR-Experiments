#!/bin/bash
# Video-averaging ReID on KITBRC1 ground-range (one mean-pooled feature per video).
#   Per-crop features -> average-pool per video -> video-to-video match.
#   Reports mAP/Rank-k PER query camera (500m, 1000m) and combined, vs the fixed 380m gallery.
#   Model: Market1501 ResNet50 (all tricks), after-BN + cosine.
#
# NOTE: query side (esp. 1000m) is preliminary -- the upstream best-track heuristic is unsupervised, so the
#       selected track isn't guaranteed to be the labeled subject; numbers improve with subject-track GT.

cd "$(dirname "$0")"

python3 tools/eval_caliper_tracklet.py --config_file='configs/softmax_triplet_with_center.yml' \
    --split ground_range \
    MODEL.DEVICE_ID "('0')" \
    MODEL.PRETRAIN_CHOICE "('self')" \
    TEST.NECK_FEAT "('after')" \
    TEST.FEAT_NORM "('yes')" \
    TEST.WEIGHT "('./weights/market_resnet50_model_120_rank1_945.pth')" \
    OUTPUT_DIR "('./crosscheck_video')"
