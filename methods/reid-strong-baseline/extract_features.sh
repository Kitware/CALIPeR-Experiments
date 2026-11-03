#!/bin/bash
# Extract BoT features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname "${BASH_SOURCE[0]}")"

SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=bot
SOURCE=market1501
WEIGHT=./weights/market_resnet50_model_120_rank1_945.pth
MAX_PER_VIDEO=0   # frames-per-tracklet cap: <=N evenly-spaced crops per video before pooling (0 = use all)
METRIC=cosine     # distance for ranking: cosine | euclidean
POOLING=video     # match unit: video (1 feat/clip, fast) | image (per crop -> pair with MAX_PER_VIDEO=16)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"
FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE',dataset='$DATASET'))")

if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
    echo "== Step 1: extracting features (GPU) -> $FEATURES_DIR =="
    python3 tools/extract_features.py --config_file='configs/softmax_triplet_with_center.yml' \
        --split "$SPLIT" --dataset "$DATASET" --method "$METHOD" --source "$SOURCE" \
        --out-dir "$FEATURES_DIR" \
        MODEL.DEVICE_ID "('0')" \
        MODEL.PRETRAIN_CHOICE "('self')" \
        TEST.NECK_FEAT "('after')" \
        TEST.WEIGHT "('$WEIGHT')"
else
    echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
fi

echo
echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
