#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
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
FEATURES_DIR=$(python3 -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

python3 -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$WEIGHT" --asset configs/softmax_triplet_with_center.yml "${FORCE[@]}" -- \
        python3 tools/extract_features.py --config_file='configs/softmax_triplet_with_center.yml' \
        --split "$SPLIT" --dataset "$DATASET" --method "$METHOD" --source "$SOURCE" \
        --out-dir '{output}' \
        MODEL.DEVICE_ID "('0')" \
        MODEL.PRETRAIN_CHOICE "('self')" \
        TEST.NECK_FEAT "('after')" \
        TEST.WEIGHT "('$WEIGHT')"

echo
echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
python3 -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
    --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
