#!/bin/bash
set -euo pipefail
FORCE=()
case "${1:-}" in
    "") ;;
    --force) FORCE=(--force) ;;
    *) echo "Usage: $0 [--force]" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Usage: $0 [--force]" >&2; exit 2; fi
# Extract CAL features for a CALIPeR split and score gallery retrieval.
# Set SPLIT to the crop split directory name. Use --force to rebuild features.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$(dirname "$0")"

PY="${PY:-python3}"                            # override with the python of an environment with CAL's dependencies
SPLIT="${SPLIT:-paperGR}"
DATASET="${DATASET:-combined_brc1_brc2}"
METHOD=cal
METRIC=cosine                                  # cosine | euclidean
MAX_PER_VIDEO=0                                # 0 = all frames, matching the other image methods
POOLING=video  # Score one feature per video for the paper.

# The paper uses the LaST checkpoint. Each entry pairs a training dataset with its weights.

COMBOS=(
    "last:weights/CAL/last-checkpoint.pth.tar"
)

export PYTHONPATH="${CALIPER_EVAL_HOME:-$REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # a single GPU (or request one from your scheduler)

for combo in "${COMBOS[@]}"; do
    SOURCE="${combo%%:*}"
    WEIGHT="${combo#*:}"

    echo "############################################################"
    echo "# CAL  source=$SOURCE  weight=$WEIGHT  split=$SPLIT  dataset=$DATASET"
    echo "############################################################"
    FEATURES_DIR=$("$PY" -c 'import sys; from caliper_eval import paths; print(paths.features_dir(*sys.argv[1:4], dataset=sys.argv[4]))' "$SPLIT" "$METHOD" "$SOURCE" "$DATASET")

    "$PY" -m caliper_eval.extraction --method "$METHOD" --source "$SOURCE" \
        --split "$SPLIT" --dataset "$DATASET" --weight "$WEIGHT" "${FORCE[@]}" -- \
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --dataset "$DATASET" --out-dir '{output}'

    echo
    echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    "$PY" -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
