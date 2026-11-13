#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
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
FEATURES_DIR=$(python3 -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

python3 -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$CKPT_DIR/model.pth" --asset "$CKPT_DIR/model.yaml" --asset "src/aligners/configs/$ALIGNER" --asset "$HOME/.cache/yolov8n-pose.pt" \
        --asset src/aligners/keypoint_predictor/pretrained_models/aligners/dfa_mobilenetv4_medium/mobilenetv4_Final.pth "${FORCE[@]}" -- \
        python3 extract_caliper_features.py --ckpt-dir "$CKPT_DIR" --source "$SOURCE" --aligner "$ALIGNER" \
        --split "$SPLIT" --dataset "$DATASET" --out-dir '{output}'

echo
echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
