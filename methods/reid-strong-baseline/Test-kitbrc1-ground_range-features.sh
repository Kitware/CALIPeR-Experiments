#!/bin/bash
# Feature-cache pipeline for BoT (Market1501 ResNet50) on the CALIPeR KITBRC1 benchmark.
#   Step 1 (GPU, once): extract RAW after-BN features -> gallery_<range>.npz / query_<range>.npz
#                       (shared caliper_eval .npz format, so the same eval scores every method fairly).
#   Step 2 (numpy):     shared eval (image + video, per range + combined) via caliper_eval.
# Paths come from caliper_eval.paths (override the whole tree with REID_BENCH_ROOT). Skips step 1 if the
# cache already exists; pass --force to re-extract.

cd "$(dirname "$0")"

SPLIT=ground_range_sim
METHOD=bot
SOURCE=market1501
WEIGHT=./weights/market_resnet50_model_120_rank1_945.pth
MAX_PER_VIDEO=0   # frames-per-tracklet cap: <=N evenly-spaced crops per video before pooling (0 = use all)
METRIC=cosine     # distance for ranking: cosine | euclidean
POOLING=video     # match unit: video (1 feat/clip, fast) | image (per crop -> pair with MAX_PER_VIDEO=16)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../" && pwd)}:$PYTHONPATH"
FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
    echo "== Step 1: extracting features (GPU) -> $FEATURES_DIR =="
    python3 tools/extract_features.py --config_file='configs/softmax_triplet_with_center.yml' \
        --split "$SPLIT" --method "$METHOD" --source "$SOURCE" --out-dir "$FEATURES_DIR" \
        MODEL.DEVICE_ID "('0')" \
        MODEL.PRETRAIN_CHOICE "('self')" \
        TEST.NECK_FEAT "('after')" \
        TEST.WEIGHT "('$WEIGHT')"
else
    echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
fi

echo
echo "== Step 2: shared eval (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
