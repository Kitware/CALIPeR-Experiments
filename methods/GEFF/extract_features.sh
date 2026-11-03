#!/bin/bash
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
    if [ ! -f "$WEIGHT" ]; then
        echo "  WEIGHT not found: $WEIGHT  -- skipping $SOURCE"; continue
    fi
    FEATURES_DIR=$("$PY" -c "from caliper_eval import paths; print(paths.features_dir('$SPLIT','$METHOD','$SOURCE',dataset='$DATASET'))")

    if [ "$1" == "--force" ] || ! ls "$FEATURES_DIR"/gallery_*.npz >/dev/null 2>&1; then
        echo "== Step 1: extracting CAL per-crop features (GPU) -> $FEATURES_DIR =="
        "$PY" extract_caliper_features.py --weight "$WEIGHT" --source "$SOURCE" --split "$SPLIT" \
            --dataset "$DATASET" --out-dir "$FEATURES_DIR" || { echo "extract FAILED for $SOURCE"; continue; }
    else
        echo "== Step 1: skipped (cache exists in $FEATURES_DIR; pass --force to re-extract) =="
    fi

    echo
    echo "== Retrieval scores (pooling=$POOLING; metric=$METRIC; per range + combined; max-per-video=$MAX_PER_VIDEO) =="
    "$PY" -m caliper_eval.eval_from_features --features-dir "$FEATURES_DIR" \
        --pooling $POOLING --metric "$METRIC" --max-per-video "$MAX_PER_VIDEO"
    echo
done
