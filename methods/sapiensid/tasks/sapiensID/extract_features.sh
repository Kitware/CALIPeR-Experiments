#!/bin/bash
# Extract SapiensID features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.
# The keypoint aligner uses YOLO pose and DFA weights in addition to the WebBody checkpoint.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$(dirname "$0")"

SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=sapiensid
SOURCE=wb12m
CKPT_DIR=../../checkpoints/sapiensid_wb12m   # has model.yaml + model.pth
ALIGNER=yolo_dfa.yaml                        # keypoint aligner (repo default for ReID)
MAX_PER_VIDEO=0                              # frames-per-tracklet cap (0 = use all); e.g. 16 with image pooling
METRIC=cosine                               # cosine | euclidean
POOLING=video                               # video (1 feat/clip) | image (per crop -> pair with MAX_PER_VIDEO=16)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"
FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE',dataset='$DATASET'))")

if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
    echo "== Step 1: extracting SapiensID features (GPU) -> $FEATURES_DIR =="
    python3 extract_caliper_features.py --ckpt-dir "$CKPT_DIR" --source "$SOURCE" --aligner "$ALIGNER" \
        --split "$SPLIT" --dataset "$DATASET" --out-dir "$FEATURES_DIR"
else
    echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
fi

echo
echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
