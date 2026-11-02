#!/bin/bash
# SapiensID feature extraction + shared eval on the CALIPeR KITBRC1 ground-range benchmark.
# Run from tasks/sapiensID/ in an env with this repo's dependencies. SapiensID is keypoint-conditioned, so it needs the
# yolo_dfa aligner (ultralytics YOLO pose @ ~/.cache/yolov8n-pose.pt + a DFA face detector). Source label =
# the WebBody checkpoint (wb12m / wb4m), since SapiensID has no Market/MSMT variant. Pick GPU via CUDA_VISIBLE_DEVICES.

cd "$(dirname "$0")"

SPLIT=ground_range
METHOD=sapiensid
SOURCE=wb12m
CKPT_DIR=../../checkpoints/sapiensid_wb12m   # has model.yaml + model.pth
ALIGNER=yolo_dfa.yaml                        # keypoint aligner (repo default for ReID)
MAX_PER_VIDEO=0                              # frames-per-tracklet cap (0 = use all); 16 for image crosscheck
METRIC=cosine                               # cosine | euclidean
POOLING=video                               # video (1 feat/clip) | image (per crop -> pair with MAX_PER_VIDEO=16)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$(cd "../../../../" && pwd)}:$PYTHONPATH"
FEATURES_DIR=$(python3 -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE'))")

if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
    echo "== Step 1: extracting SapiensID features (GPU) -> $FEATURES_DIR =="
    python3 extract_caliper_features.py --ckpt-dir "$CKPT_DIR" --source "$SOURCE" --aligner "$ALIGNER" \
        --split "$SPLIT" --out-dir "$FEATURES_DIR"
else
    echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
fi

echo
echo "== Step 2: shared eval (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
